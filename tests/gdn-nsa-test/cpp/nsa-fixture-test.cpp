// Level-1 NSA operator validation: ggml_nsa_select / ggml_nsa_sel_attn(_ext) + the model's compression/gating
// graph vs fla reference outputs on identical inputs (fixtures from reference/make_nsa_fixtures.py).
// For every fixture x backend (CPU, CUDA) x K/V type (F32, F16):
//   (a) sel      : k_cmp = block mean of K (ggml ops, as build_layer_nsa), idx = ggml_nsa_select; set match vs fla
//   (b) slc_ref  : ggml_nsa_sel_attn with the fla indices                      vs fla o_slc
//   (c) slc_own  : ggml_nsa_sel_attn with our indices                          vs fla o_slc (gated on rows whose set matched)
//   (d) swa      : ggml_nsa_sel_attn_ext, swa_idx as the model, window 512     vs fla o_swa (+ rows p >= 512: truncated window)
//   (e) cmp      : dense attention over block means + zero block (model graph)  vs fla o_cmp
//   (f) o        : sigmoid-gated sum (model graph)                              vs fla naive_nsa o; one-hot gates vs branches
// usage: nsa-fixture-test [fixture_dir] [out_json] [case ...]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static const int D = 64, Hq = 32, Hkv = 2, G = Hq / Hkv, BS = 64, S = 16, W = 512;
static const int NW = (W + BS - 2) / BS + 1; // blocks overlapping the window (model: n_win_blk)

// thresholds: {max abs, NRMSE}. F32 from the task spec; F16 justified from the measured numbers (see report)
struct thr { double atol, nrmse; };
static const thr THR_F32 = { 1e-4, 1e-3 };
static const thr THR_F16 = { 4e-3, 1e-3 };
// the compression branch (and the gated sums containing it) is built from ggml_mul_mat; ggml-cuda creates its cuBLAS
// handles with CUBLAS_TF32_TENSOR_OP_MATH (common.cuh), so on CUDA F32 GEMMs run in TF32 (10-bit mantissa,
// u = 2^-11 ~ 4.9e-4). That class gets {5e-4, 1e-3}; NVIDIA_TF32_OVERRIDE=0 disables TF32 and restores the F32 class.
static const thr THR_TF32 = { 5e-4, 1e-3 };
// a selection mismatch is explained if the fla 16th-vs-17th score margin (relative to the 16th score) is below this
static const double SEL_TOL_F32 = 1e-4, SEL_TOL_F16 = 1e-3;
static bool g_tf32 = true; // set in main from NVIDIA_TF32_OVERRIDE

struct fixture {
    std::string name;
    int T = 0, NBc = 0, det = 0;
    std::vector<float> q, k, v, g, kcmp, vcmp, scores, margin, o_cmp, o_slc, o_swa, o;
    std::vector<int32_t> idx;
};

static double json_num(const std::string & s, const char * key) {
    const std::string k = std::string("\"") + key + "\":";
    size_t p = s.find(k);
    if (p == std::string::npos) { fprintf(stderr, "missing key %s\n", key); exit(2); }
    return strtod(s.c_str() + p + k.size(), nullptr);
}

template <typename T> static std::vector<T> load(const std::string & path, size_t n) {
    std::vector<T> x(n);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    size_t r = fread(x.data(), sizeof(T), n, f);
    if (r != n || fgetc(f) != EOF) { fprintf(stderr, "%s: size mismatch (want %zu)\n", path.c_str(), n); exit(2); }
    fclose(f);
    return x;
}

static fixture load_fixture(const std::string & dir, const std::string & name) {
    std::ifstream jf(dir + "/" + name + ".json");
    std::stringstream ss; ss << jf.rdbuf();
    std::string js = ss.str();
    js.erase(std::remove_if(js.begin(), js.end(), [](char c) { return c == ' ' || c == '\n'; }), js.end());
    fixture f;
    f.name = name;
    f.T = (int) json_num(js, "T");
    f.NBc = (int) json_num(js, "NBc");
    f.det = (int) json_num(js, "deterministic");
    if ((int) json_num(js, "Hq") != Hq || (int) json_num(js, "Hkv") != Hkv || (int) json_num(js, "D") != D ||
        (int) json_num(js, "block_size") != BS || (int) json_num(js, "n_sel") != S || (int) json_num(js, "window") != W) {
        fprintf(stderr, "%s: shape mismatch\n", name.c_str()); exit(2);
    }
    const size_t T = f.T, NBc = f.NBc;
    auto p = [&](const char * t) { return dir + "/" + name + "." + t + ".bin"; };
    f.q = load<float>(p("q"), T * Hq * D);
    f.k = load<float>(p("k"), T * Hkv * D);
    f.v = load<float>(p("v"), T * Hkv * D);
    f.g = load<float>(p("g"), T * Hq * 3);
    f.kcmp = load<float>(p("kcmp"), NBc * Hkv * D);
    f.vcmp = load<float>(p("vcmp"), NBc * Hkv * D);
    f.scores = load<float>(p("scores"), T * Hkv * NBc);
    f.idx = load<int32_t>(p("idx"), T * Hkv * S);
    f.margin = load<float>(p("margin"), T * Hkv);
    f.o_cmp = load<float>(p("o_cmp"), T * Hq * D);
    f.o_slc = load<float>(p("o_slc"), T * Hq * D);
    f.o_swa = load<float>(p("o_swa"), T * Hq * D);
    f.o = load<float>(p("o"), T * Hq * D);
    return f;
}

struct metrics {
    long n = 0, nonfinite = 0;
    double maxabs = 0, rmse = 0, nrmse = 0, cos = 1;
};

// x = reference, y = ours; rows of `row` elements, only rows with mask[r] != 0 (mask may be empty = all)
static metrics compare(const float * x, const float * y, size_t nel, size_t row = 1, const std::vector<char> & mask = {}) {
    metrics m;
    double dd = 0, xx = 0, yy = 0, xy = 0;
    for (size_t i = 0; i < nel; ++i) {
        if (!mask.empty() && !mask[i / row]) continue;
        m.n++;
        if (!std::isfinite(y[i])) { m.nonfinite++; continue; }
        const double d = (double) y[i] - x[i];
        m.maxabs = std::max(m.maxabs, std::fabs(d));
        dd += d * d; xx += (double) x[i] * x[i]; yy += (double) y[i] * y[i]; xy += (double) x[i] * y[i];
    }
    m.rmse = m.n ? std::sqrt(dd / m.n) : 0;
    m.nrmse = xx > 0 ? std::sqrt(dd / xx) : (dd > 0 ? INFINITY : 0);
    m.cos = (xx > 0 && yy > 0) ? xy / std::sqrt(xx * yy) : (xx == 0 && yy == 0 ? 1 : 0);
    return m;
}

struct record {
    std::string cs, be, kv, br;
    metrics m;
    bool has_m = true, pass = true;
    std::string thr_class = "f32";
    double exact = -1; long mism = 0, unexplained = 0; double worst_margin = 0;
};

static std::vector<record> g_rec;
static bool g_fail = false;

static void add(const fixture & f, const char * be, const char * kv, const char * br, const metrics & m, const thr & t) {
    record r{f.name, be, kv, br, m};
    r.thr_class = t.atol == THR_TF32.atol && t.nrmse == THR_TF32.nrmse ? "tf32" : t.atol == THR_F16.atol ? "f16" : t.atol == THR_F32.atol ? "f32" : "custom";
    r.pass = m.nonfinite == 0 && m.maxabs <= t.atol && m.nrmse <= t.nrmse;
    if (m.n == 0) r.pass = true;
    g_fail |= !r.pass;
    printf("%-10s %-6s %-4s %-12s n=%9ld max_abs=%.3e rmse=%.3e nrmse=%.3e cos=%.8f nonfinite=%ld [%s] %s\n", f.name.c_str(), be, kv,
           br, m.n, m.maxabs, m.rmse, m.nrmse, m.cos, m.nonfinite, r.thr_class.c_str(), r.pass ? "OK" : "FAIL");
    g_rec.push_back(r);
}

static void run(ggml_backend_t backend, const fixture & f, ggml_type kvt) {
    const int T = f.T;
    const int N = ((T + 255) / 256) * 256; // KV cache size: multiple of 256 >= T, rows >= T hold garbage
    const int NB = N / BS;
    const float scale = 1.0f / sqrtf((float) D);
    const char * be = ggml_backend_name(backend);
    const char * kv = ggml_type_name(kvt);
    const thr th = kvt == GGML_TYPE_F16 ? THR_F16 : THR_F32;
    // cuBLAS-backed branches (e_cmp, gated sums): TF32 on CUDA unless disabled; F16 K/V keeps the (looser) F16 class
    const thr th_mm = (!ggml_backend_is_cpu(backend) && g_tf32 && kvt != GGML_TYPE_F16) ? THR_TF32 : th;

    ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead_custom(2048, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tq   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * tk   = ggml_new_tensor_3d(ctx, kvt, D, Hkv, N);
    ggml_tensor * tv   = ggml_new_tensor_3d(ctx, kvt, D, Hkv, N);
    ggml_tensor * tpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * tidx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, Hkv, T);
    ggml_tensor * tswa = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, NW, Hkv, T);
    ggml_tensor * tcm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NB + 1, T);
    ggml_tensor * tgl  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, Hq, T);
    ggml_tensor * t1h[3];
    for (auto & t : t1h) t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, Hq, T);

    // ---- graph, following build_layer_nsa ----
    auto block_mean = [&](ggml_tensor * x) {
        const int64_t dx = x->ne[0], h = x->ne[1];
        ggml_tensor * xf = ggml_cast(ctx, x, GGML_TYPE_F32);
        xf = ggml_reshape_3d(ctx, xf, dx * h, BS, NB);
        xf = ggml_cont(ctx, ggml_permute(ctx, xf, 1, 0, 2, 3));
        xf = ggml_mean(ctx, xf);
        return ggml_reshape_3d(ctx, xf, dx, h, NB);
    };
    auto dense_attn = [&](ggml_tensor * q, ggml_tensor * kk, ggml_tensor * v_t, ggml_tensor * mask) {
        ggml_tensor * kq = ggml_mul_mat(ctx, ggml_permute(ctx, kk, 0, 2, 1, 3), ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v_t, kq);
        return ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    };
    ggml_tensor * k_cmp = block_mean(tk);
    ggml_tensor * v_cmp = block_mean(tv);
    ggml_tensor * k_cmp_z = ggml_pad(ctx, k_cmp, 0, 0, 1, 0);
    ggml_tensor * v_cmp_z = ggml_pad(ctx, v_cmp, 0, 0, 1, 0);
    ggml_tensor * o_cmp = dense_attn(tq, k_cmp_z, ggml_cont(ctx, ggml_permute(ctx, v_cmp_z, 1, 2, 0, 3)), tcm);
    ggml_tensor * idx_own = ggml_nsa_select(ctx, tq, k_cmp, tpos, BS, S, scale);
    ggml_tensor * o_slc_ref = ggml_nsa_sel_attn(ctx, tq, tk, tv, tidx, tpos, BS, scale);
    ggml_tensor * o_slc = ggml_nsa_sel_attn(ctx, tq, tk, tv, idx_own, tpos, BS, scale);
    ggml_tensor * o_swa = ggml_nsa_sel_attn_ext(ctx, tq, tk, tv, tswa, tpos, BS, scale, W);
    auto gated = [&](ggml_tensor * gates) {
        auto gate = [&](int i) {
            return ggml_cont(ctx, ggml_view_3d(ctx, gates, 1, Hq, T, gates->nb[1], gates->nb[2], i * ggml_element_size(gates)));
        };
        ggml_tensor * o = ggml_mul(ctx, o_cmp, gate(0));
        o = ggml_add(ctx, o, ggml_mul(ctx, o_slc, gate(1)));
        return ggml_add(ctx, o, ggml_mul(ctx, o_swa, gate(2)));
    };
    ggml_tensor * o = gated(ggml_sigmoid(ctx, tgl));
    ggml_tensor * o1h[3];
    for (int i = 0; i < 3; ++i) o1h[i] = gated(t1h[i]);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);
    for (ggml_tensor * t : { k_cmp, v_cmp, o_cmp, idx_own, o_slc_ref, o_slc, o_swa, o, o1h[0], o1h[1], o1h[2] }) {
        ggml_build_forward_expand(gf, t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // ---- inputs ----
    std::vector<float> kf((size_t) N * Hkv * D), vf((size_t) N * Hkv * D);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (auto & x : kf) x = nd(rng);
    for (auto & x : vf) x = nd(rng);
    std::copy(f.k.begin(), f.k.end(), kf.begin());
    std::copy(f.v.begin(), f.v.end(), vf.begin());
    if (kvt == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> k16(kf.size()), v16(vf.size());
        ggml_fp32_to_fp16_row(kf.data(), k16.data(), kf.size());
        ggml_fp32_to_fp16_row(vf.data(), v16.data(), vf.size());
        ggml_backend_tensor_set(tk, k16.data(), 0, ggml_nbytes(tk));
        ggml_backend_tensor_set(tv, v16.data(), 0, ggml_nbytes(tv));
    } else {
        ggml_backend_tensor_set(tk, kf.data(), 0, ggml_nbytes(tk));
        ggml_backend_tensor_set(tv, vf.data(), 0, ggml_nbytes(tv));
    }
    ggml_backend_tensor_set(tq, f.q.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tgl, f.g.data(), 0, ggml_nbytes(tgl));
    ggml_backend_tensor_set(tidx, f.idx.data(), 0, ggml_nbytes(tidx));
    std::vector<int32_t> pos(T), swa((size_t) NW * Hkv * T);
    for (int t = 0; t < T; ++t) pos[t] = t;
    ggml_backend_tensor_set(tpos, pos.data(), 0, ggml_nbytes(tpos));
    for (int t = 0; t < T; ++t) {
        const int cur = pos[t] / BS;
        for (int h = 0; h < Hkv; ++h) {
            for (int w = 0; w < NW; ++w) swa[((size_t) t * Hkv + h) * NW + w] = cur - w >= 0 ? cur - w : -1;
        }
    }
    ggml_backend_tensor_set(tswa, swa.data(), 0, ggml_nbytes(tswa));
    std::vector<float> cm((size_t) (NB + 1) * T, -INFINITY);
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < NB && (b + 1) * BS - 1 <= pos[t]; ++b) cm[(size_t) t * (NB + 1) + b] = 0.0f;
        if (pos[t] < BS - 1) cm[(size_t) t * (NB + 1) + NB] = 0.0f;
    }
    ggml_backend_tensor_set(tcm, cm.data(), 0, ggml_nbytes(tcm));
    for (int i = 0; i < 3; ++i) {
        std::vector<float> g1((size_t) 3 * Hq * T, 0.0f);
        for (size_t r = 0; r < (size_t) Hq * T; ++r) g1[r * 3 + i] = 1.0f;
        ggml_backend_tensor_set(t1h[i], g1.data(), 0, ggml_nbytes(t1h[i]));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }

    auto get = [](ggml_tensor * t) {
        std::vector<float> x(ggml_nelements(t));
        ggml_backend_tensor_get(t, x.data(), 0, ggml_nbytes(t));
        return x;
    };
    const size_t no = (size_t) T * Hq * D;

    // (a) selection
    {
        const int nbc = T / BS; // complete blocks
        std::vector<float> kc = get(k_cmp);
        metrics mk = compare(f.kcmp.data(), kc.data(), (size_t) nbc * Hkv * D);
        add(f, be, kv, "kcmp", mk, kvt == GGML_TYPE_F16 ? thr{ 2e-3, 2e-3 } : thr{ 1e-5, 1e-5 });
        std::vector<int32_t> si(ggml_nelements(idx_own));
        ggml_backend_tensor_get(idx_own, si.data(), 0, ggml_nbytes(idx_own));
        long match = 0, mism = 0, unexpl = 0;
        double worst = 0, worst_any = 0;
        std::vector<char> row_ok((size_t) T * Hkv, 1);
        const double tol = kvt == GGML_TYPE_F16 ? SEL_TOL_F16 : SEL_TOL_F32;
        for (int t = 0; t < T; ++t) {
            const int cur = t / BS;
            for (int h = 0; h < Hkv; ++h) {
                const size_t r = (size_t) t * Hkv + h;
                std::vector<int32_t> a(&si[r * S], &si[r * S] + S), b(&f.idx[r * S], &f.idx[r * S] + S);
                std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                if (a == b) { match++; continue; }
                mism++; row_ok[r] = 0;
                // relative margin of the fla scores: (s16 - s17) / s16 over the candidates
                std::vector<float> c;
                for (int bb = 0; bb < f.NBc; ++bb) {
                    const bool cand = (bb + 1) * BS - 1 <= t || bb == 0 || bb == cur || bb == cur - 1;
                    if (cand) c.push_back(f.scores[r * f.NBc + bb]);
                }
                std::sort(c.rbegin(), c.rend());
                const double rel = c.size() > (size_t) S ? (c[S - 1] - c[S]) / c[S - 1] : INFINITY;
                worst_any = std::max(worst_any, rel);
                if (!(rel < tol) || f.det) { unexpl++; worst = std::max(worst, rel); }
            }
        }
        record rr{f.name, be, kv, "sel"};
        rr.has_m = false;
        rr.exact = 100.0 * match / ((double) T * Hkv);
        rr.mism = mism; rr.unexplained = unexpl; rr.worst_margin = worst_any;
        rr.pass = unexpl == 0 && (!f.det || mism == 0);
        g_fail |= !rr.pass;
        printf("%-10s %-6s %-4s %-12s rows=%ld exact=%.4f%% mismatch=%ld max_rel_margin_of_mismatch=%.3e unexplained=%ld %s\n",
               f.name.c_str(), be, kv, "sel", (long) T * Hkv, rr.exact, mism, worst_any, unexpl, rr.pass ? "OK" : "FAIL");
        g_rec.push_back(rr);

        std::vector<char> qrow((size_t) T * Hq);
        for (int t = 0; t < T; ++t) for (int hq = 0; hq < Hq; ++hq) qrow[(size_t) t * Hq + hq] = row_ok[(size_t) t * Hkv + hq / G];

        // (b) (c)
        std::vector<float> x = get(o_slc_ref);
        add(f, be, kv, "b_slc_refidx", compare(f.o_slc.data(), x.data(), no), th);
        x = get(o_slc);
        metrics mall = compare(f.o_slc.data(), x.data(), no);
        if (mism == 0) {
            add(f, be, kv, "c_slc_ownidx", mall, th);
        } else {
            printf("%-10s %-6s %-4s %-12s (all rows, info) max_abs=%.3e nrmse=%.3e\n", f.name.c_str(), be, kv, "c_slc_ownidx", mall.maxabs, mall.nrmse);
            add(f, be, kv, "c_slc_own_m", compare(f.o_slc.data(), x.data(), no, D, qrow), th);
        }
        // (d)
        x = get(o_swa);
        add(f, be, kv, "d_swa", compare(f.o_swa.data(), x.data(), no), th);
        if (T > W) {
            std::vector<char> trunc((size_t) T * Hq, 0);
            for (int t = W; t < T; ++t) for (int hq = 0; hq < Hq; ++hq) trunc[(size_t) t * Hq + hq] = 1;
            add(f, be, kv, "d_swa_p>=512", compare(f.o_swa.data(), x.data(), no, D, trunc), th);
        }
        // (e)
        x = get(o_cmp);
        add(f, be, kv, "e_cmp", compare(f.o_cmp.data(), x.data(), no), th_mm);
        // (f)
        x = get(o);
        if (mism == 0) {
            add(f, be, kv, "f_o", compare(f.o.data(), x.data(), no), th_mm);
        } else {
            metrics ma = compare(f.o.data(), x.data(), no);
            printf("%-10s %-6s %-4s %-12s (all rows, info) max_abs=%.3e nrmse=%.3e\n", f.name.c_str(), be, kv, "f_o", ma.maxabs, ma.nrmse);
            add(f, be, kv, "f_o_m", compare(f.o.data(), x.data(), no, D, qrow), th_mm);
        }
        const std::vector<float> * refs[3] = { &f.o_cmp, &f.o_slc, &f.o_swa };
        const char * nm[3] = { "f_1h_cmp", "f_1h_slc", "f_1h_swa" };
        for (int i = 0; i < 3; ++i) {
            x = get(o1h[i]);
            add(f, be, kv, nm[i], compare(refs[i]->data(), x.data(), no, D, i == 1 && mism ? qrow : std::vector<char>{}), i == 0 ? th_mm : th);
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : "fixtures/nsa";
    const std::string out = argc > 2 ? argv[2] : "l1_nsa.json";
    std::vector<std::string> cases;
    for (int i = 3; i < argc; ++i) cases.push_back(argv[i]);
    if (cases.empty()) {
        std::ifstream cf(dir + "/cases.txt");
        for (std::string l; std::getline(cf, l);) if (!l.empty()) cases.push_back(l);
    }
    std::vector<fixture> fx;
    for (auto & c : cases) fx.push_back(load_fixture(dir, c));

    { const char * e = getenv("NVIDIA_TF32_OVERRIDE"); g_tf32 = !(e && atoi(e) == 0); }
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, 32);
        for (auto & f : fx) {
            for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16 }) run(be, f, t);
        }
        ggml_backend_free(be);
    }

    FILE * jf = fopen(out.c_str(), "w");
    if (!jf) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 2; }
    fprintf(jf, "{\n \"thresholds\": {\"F32\": {\"max_abs\": %g, \"nrmse\": %g, \"sel_rel_margin\": %g}, "
                "\"F16\": {\"max_abs\": %g, \"nrmse\": %g, \"sel_rel_margin\": %g}, \"TF32\": {\"max_abs\": %g, \"nrmse\": %g}},\n"
                " \"cuda_tf32\": %s,\n \"pass\": %s,\n \"results\": [\n",
            THR_F32.atol, THR_F32.nrmse, SEL_TOL_F32, THR_F16.atol, THR_F16.nrmse, SEL_TOL_F16, THR_TF32.atol, THR_TF32.nrmse,
            g_tf32 ? "true" : "false", g_fail ? "false" : "true");
    auto num = [](double x) { static char b[64]; if (std::isfinite(x)) snprintf(b, sizeof b, "%.6e", x); else snprintf(b, sizeof b, "null"); return std::string(b); };
    for (size_t i = 0; i < g_rec.size(); ++i) {
        const record & r = g_rec[i];
        fprintf(jf, "  {\"case\": \"%s\", \"backend\": \"%s\", \"kv\": \"%s\", \"branch\": \"%s\", \"pass\": %s", r.cs.c_str(),
                r.be.c_str(), r.kv.c_str(), r.br.c_str(), r.pass ? "true" : "false");
        if (r.has_m) {
            fprintf(jf, ", \"thr\": \"%s\"", r.thr_class.c_str());
            fprintf(jf, ", \"n\": %ld, \"max_abs\": %s, \"rmse\": %s, \"nrmse\": %s, \"cos\": %s, \"nonfinite\": %ld", r.m.n,
                    num(r.m.maxabs).c_str(), num(r.m.rmse).c_str(), num(r.m.nrmse).c_str(), num(r.m.cos).c_str(), r.m.nonfinite);
        } else {
            fprintf(jf, ", \"exact_pct\": %.6f, \"mismatch_rows\": %ld, \"unexplained\": %ld, \"max_rel_margin_of_mismatch\": %s",
                    r.exact, r.mism, r.unexplained, num(r.worst_margin).c_str());
        }
        fprintf(jf, "}%s\n", i + 1 < g_rec.size() ? "," : "");
    }
    fprintf(jf, " ]\n}\n");
    fclose(jf);
    printf("%s -> %s\n", g_fail ? "FAIL" : "PASS", out.c_str());
    return g_fail ? 1 : 0;
}
