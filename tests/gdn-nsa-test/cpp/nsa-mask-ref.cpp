// Cross-checks the two NSA ops against reference paths built only from existing ggml ops, on every backend:
//  1. ggml_nsa_sel_attn vs dense attention with a host-built mask that unmasks exactly the selected tokens
//     (the DSA-style "top-k -> mask -> dense attention" path), one dense call per kv head.
//  2. the same op with V passed as a transposed view (the V cache layout with flash attention off) vs the same reference.
//  3. ggml_nsa_sel_attn_ext over the blocks that overlap the window (the model's sparse sliding window) vs dense
//     attention with a sliding-window mask, which does not depend on the block indices.
//  4. ggml_nsa_select vs mul_mat + soft_max(mask of complete blocks) + mean over the group + forced-block
//     bias + ggml_top_k, and the select output of every backend bit for bit vs the CPU backend.
//  5. quantized K/V cache (q8_0, q4_0): ggml_nsa_sel_attn(_ext) on the quantized K/V vs the same op on the
//     host-dequantized F32 values (the kernels must only differ in how they load K/V), and ggml_cast to F32
//     (the block mean input) vs the host dequantization.
// usage: nsa-mask-ref [T] [N]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

static const int D = 64, Hq = 32, Hkv = 2, G = Hq / Hkv, BS = 64, S = 16;
static const int W  = 512;                     // sliding window of the model
static const int SW = (W + BS - 2) / BS + 1;   // blocks that can overlap the window (9)

struct run_result {
    std::vector<float> o_op, o_op_t, o_ref;     // sel_attn, sel_attn with transposed V, dense reference
    std::vector<float> o_swa_op, o_swa_ref;     // sel_attn_ext over the window blocks, dense sliding window
    std::vector<int32_t> sel_op, sel_ref;
    std::vector<float> score_ref; // [T, NB] per kv head h (stored [h][t][b])
};

static run_result run(ggml_backend_t backend, int T, int N, const std::vector<float> & q, const std::vector<ggml_fp16_t> & k16,
                      const std::vector<ggml_fp16_t> & v16, const std::vector<int32_t> & idx, const std::vector<int32_t> & pos,
                      const std::vector<float> & kcmp) {
    const int NB = N / BS;
    const float scale = 1.0f / sqrtf((float) D);

    ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tq    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * tk    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, Hkv, N);
    ggml_tensor * tv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, Hkv, N);
    ggml_tensor * tvt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, N, D, Hkv); // V stored [h][d][j] like the transposed V cache
    ggml_tensor * tidx  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, Hkv, T);
    ggml_tensor * tidxw = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, SW, Hkv, T);
    ggml_tensor * tpos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * tkcmp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, NB);
    ggml_tensor * tmask[Hkv], * tcmask, * tbias[Hkv], * twmask;
    for (int h = 0; h < Hkv; ++h) {
        tmask[h] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);
        tbias[h] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NB, T);
    }
    tcmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NB, T);
    twmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);

    // new ops; the transposed V view [D, Hkv, N] has nb0 = 2*N, nb1 = 2*N*D, nb2 = 2 (as from the V cache)
    ggml_tensor * o_op     = ggml_nsa_sel_attn(ctx, tq, tk, tv, tidx, tpos, BS, scale);
    ggml_tensor * o_op_t   = ggml_nsa_sel_attn(ctx, tq, tk, ggml_permute(ctx, tvt, 2, 0, 1, 3), tidx, tpos, BS, scale);
    ggml_tensor * o_swa_op = ggml_nsa_sel_attn_ext(ctx, tq, tk, tv, tidxw, tpos, BS, scale, W);
    ggml_tensor * sel_op   = ggml_nsa_select(ctx, tq, tkcmp, tpos, BS, S, scale);

    std::vector<ggml_tensor *> o_ref(Hkv), o_swa_ref(Hkv), sel_ref(Hkv), score_ref(Hkv);
    for (int h = 0; h < Hkv; ++h) {
        // q heads of group h: [D, G, T] -> [D, T, G]
        ggml_tensor * qh = ggml_view_3d(ctx, tq, D, G, T, tq->nb[1], tq->nb[2], (size_t) h * G * tq->nb[1]);
        ggml_tensor * qp = ggml_permute(ctx, qh, 0, 2, 1, 3);

        // 1.-3. dense masked attention over kv head h
        ggml_tensor * kh = ggml_view_2d(ctx, tk, D, N, tk->nb[2], h * tk->nb[1]);
        ggml_tensor * vh = ggml_view_2d(ctx, tv, D, N, tv->nb[2], h * tv->nb[1]);
        ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, ggml_cast(ctx, vh, GGML_TYPE_F32))); // [N, D]
        auto dense = [&](ggml_tensor * mask) {
            ggml_tensor * kq = ggml_mul_mat(ctx, kh, qp);                  // [N, T, G]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                 // [D, T, G]
            return ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));    // [D, G, T]
        };
        o_ref[h]     = dense(tmask[h]);
        o_swa_ref[h] = dense(twmask);

        // 4. selection from existing ops
        ggml_tensor * kc = ggml_view_2d(ctx, tkcmp, D, NB, tkcmp->nb[2], h * tkcmp->nb[1]);
        ggml_tensor * sc = ggml_mul_mat(ctx, kc, qp);                 // [NB, T, G]
        sc = ggml_soft_max_ext(ctx, sc, tcmask, scale, 0.0f);
        sc = ggml_cont(ctx, ggml_permute(ctx, sc, 1, 2, 0, 3));      // [G, NB, T]
        sc = ggml_reshape_2d(ctx, ggml_mean(ctx, sc), NB, T);         // [NB, T]
        score_ref[h] = sc;
        sel_ref[h] = ggml_top_k(ctx, ggml_add(ctx, sc, tbias[h]), S); // [S, T], in no guaranteed order
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (ggml_tensor * t : { o_op, o_op_t, o_swa_op, sel_op }) {
        ggml_build_forward_expand(gf, t);
    }
    for (int h = 0; h < Hkv; ++h) {
        ggml_build_forward_expand(gf, o_ref[h]);
        ggml_build_forward_expand(gf, o_swa_ref[h]);
        ggml_build_forward_expand(gf, sel_ref[h]);
        ggml_build_forward_expand(gf, score_ref[h]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tq, q.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk, k16.data(), 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tv, v16.data(), 0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tidx, idx.data(), 0, ggml_nbytes(tidx));
    ggml_backend_tensor_set(tpos, pos.data(), 0, ggml_nbytes(tpos));
    ggml_backend_tensor_set(tkcmp, kcmp.data(), 0, ggml_nbytes(tkcmp));
    {
        std::vector<ggml_fp16_t> vt16((size_t) N * D * Hkv);
        for (int j = 0; j < N; ++j) {
            for (int h = 0; h < Hkv; ++h) {
                for (int d = 0; d < D; ++d) {
                    vt16[((size_t) h * D + d) * N + j] = v16[((size_t) j * Hkv + h) * D + d];
                }
            }
        }
        ggml_backend_tensor_set(tvt, vt16.data(), 0, ggml_nbytes(tvt));
    }
    {
        // the blocks that can overlap the window, newest first, as the model builds them
        std::vector<int32_t> iw((size_t) SW * Hkv * T);
        for (int t = 0; t < T; ++t) {
            const int cur = pos[t] / BS;
            for (int h = 0; h < Hkv; ++h) {
                for (int w = 0; w < SW; ++w) {
                    iw[((size_t) t * Hkv + h) * SW + w] = cur - w >= 0 ? cur - w : -1;
                }
            }
        }
        ggml_backend_tensor_set(tidxw, iw.data(), 0, ggml_nbytes(tidxw));
    }

    // host masks
    std::vector<float> m((size_t) N * T), cm((size_t) NB * T), bias((size_t) NB * T);
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < NB; ++b) {
            cm[(size_t) t * NB + b] = ((b + 1) * BS - 1 <= pos[t]) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(tcmask, cm.data(), 0, ggml_nbytes(tcmask));
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            m[(size_t) t * N + j] = (j <= pos[t] && j > pos[t] - W) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(twmask, m.data(), 0, ggml_nbytes(twmask));
    for (int h = 0; h < Hkv; ++h) {
        std::fill(m.begin(), m.end(), -INFINITY);
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < S; ++s) {
                const int b = idx[((size_t) t * Hkv + h) * S + s];
                if (b < 0) continue;
                for (int j = b * BS; j < std::min(N, (b + 1) * BS); ++j) {
                    if (j <= pos[t]) m[(size_t) t * N + j] = 0.0f;
                }
            }
            const int cur = pos[t] / BS;
            for (int b = 0; b < NB; ++b) {
                const bool complete = (b + 1) * BS - 1 <= pos[t];
                const bool forced   = b == 0 || b == cur || b == cur - 1;
                bias[(size_t) t * NB + b] = forced && b <= cur ? 2.0f : (complete ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(tmask[h], m.data(), 0, ggml_nbytes(tmask[h]));
        ggml_backend_tensor_set(tbias[h], bias.data(), 0, ggml_nbytes(tbias[h]));
    }

    ggml_backend_graph_compute(backend, gf);

    auto get = [](ggml_tensor * t, std::vector<float> & out) {
        out.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    };
    run_result r;
    get(o_op, r.o_op);
    get(o_op_t, r.o_op_t);
    get(o_swa_op, r.o_swa_op);
    r.sel_op.resize(ggml_nelements(sel_op));
    ggml_backend_tensor_get(sel_op, r.sel_op.data(), 0, ggml_nbytes(sel_op));
    // assemble reference outputs in the op layouts
    r.o_ref.assign(r.o_op.size(), 0.0f);
    r.o_swa_ref.assign(r.o_op.size(), 0.0f);
    r.sel_ref.assign(r.sel_op.size(), -1);
    r.score_ref.resize((size_t) Hkv * T * NB);
    for (int h = 0; h < Hkv; ++h) {
        for (auto [src, dst] : { std::make_pair(o_ref[h], &r.o_ref), std::make_pair(o_swa_ref[h], &r.o_swa_ref) }) {
            std::vector<float> oh;
            get(src, oh);
            for (int t = 0; t < T; ++t) {
                for (int g = 0; g < G; ++g) {
                    memcpy(&(*dst)[((size_t) t * Hq + h * G + g) * D], &oh[((size_t) t * G + g) * D], D * sizeof(float));
                }
            }
        }
        std::vector<int32_t> sh(ggml_nelements(sel_ref[h]));
        ggml_backend_tensor_get(sel_ref[h], sh.data(), 0, ggml_nbytes(sel_ref[h]));
        for (int t = 0; t < T; ++t) {
            memcpy(&r.sel_ref[((size_t) t * Hkv + h) * S], &sh[(size_t) t * S], S * sizeof(int32_t));
        }
        ggml_backend_tensor_get(score_ref[h], &r.score_ref[(size_t) h * T * NB], 0, ggml_nbytes(score_ref[h]));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// normalized squared error of a against the reference b, and the max abs difference
static void nmse_of(const std::vector<float> & a, const std::vector<float> & b, double & e, double & maxd) {
    double num = 0, den = 0;
    maxd = 0;
    for (size_t j = 0; j < a.size(); ++j) {
        const double d = (double) a[j] - b[j];
        num += d * d; den += (double) b[j] * b[j]; maxd = std::max(maxd, std::fabs(d));
    }
    e = num / std::max(den, 1e-30);
}

struct quant_result {
    double nmse = 0, max_abs = 0;     // sel_attn on quantized K/V vs on the dequantized F32 K/V (window 0 and 512)
    double cast_max_abs = 0;          // ggml_cast(K/V, F32) vs host dequantization
};

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, double * nmse = nullptr) {
    double num = 0, den = 0, m = 0;
    for (size_t j = 0; j < a.size(); ++j) {
        const double d = (double) a[j] - b[j];
        num += d * d; den += (double) b[j] * b[j]; m = std::max(m, std::fabs(d));
    }
    if (nmse) *nmse = std::max(*nmse, num / std::max(den, 1e-30));
    return m;
}

static quant_result run_quant(ggml_backend_t backend, ggml_type type, int T, int N, const std::vector<float> & q,
                              const std::vector<float> & k32, const std::vector<float> & v32, const std::vector<int32_t> & idx,
                              const std::vector<int32_t> & pos) {
    const float scale = 1.0f / sqrtf((float) D);
    const int64_t nrows = (int64_t) Hkv * N;

    // host quantization of every (token, kv head) row of D values, and its dequantization
    const size_t row_size = ggml_row_size(type, D);
    std::vector<uint8_t> kq(row_size * nrows), vq(row_size * nrows);
    ggml_quantize_chunk(type, k32.data(), kq.data(), 0, nrows, D, nullptr);
    ggml_quantize_chunk(type, v32.data(), vq.data(), 0, nrows, D, nullptr);
    std::vector<float> kd((size_t) D * nrows), vd((size_t) D * nrows);
    ggml_get_type_traits(type)->to_float(kq.data(), kd.data(), (int64_t) D * nrows);
    ggml_get_type_traits(type)->to_float(vq.data(), vd.data(), (int64_t) D * nrows);

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tq   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * tkq  = ggml_new_tensor_3d(ctx, type, D, Hkv, N);
    ggml_tensor * tvq  = ggml_new_tensor_3d(ctx, type, D, Hkv, N);
    ggml_tensor * tkd  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
    ggml_tensor * tvd  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
    ggml_tensor * tidx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, Hkv, T);
    ggml_tensor * tpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);

    ggml_tensor * oq   = ggml_nsa_sel_attn(ctx, tq, tkq, tvq, tidx, tpos, BS, scale);
    ggml_tensor * od   = ggml_nsa_sel_attn(ctx, tq, tkd, tvd, tidx, tpos, BS, scale);
    ggml_tensor * oqw  = ggml_nsa_sel_attn_ext(ctx, tq, tkq, tvq, tidx, tpos, BS, scale, 512);
    ggml_tensor * odw  = ggml_nsa_sel_attn_ext(ctx, tq, tkd, tvd, tidx, tpos, BS, scale, 512);
    ggml_tensor * kc   = ggml_cast(ctx, tkq, GGML_TYPE_F32);
    ggml_tensor * vc   = ggml_cast(ctx, tvq, GGML_TYPE_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : { oq, od, oqw, odw, kc, vc }) ggml_build_forward_expand(gf, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(tq,   q.data(),   0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tkq,  kq.data(),  0, ggml_nbytes(tkq));
    ggml_backend_tensor_set(tvq,  vq.data(),  0, ggml_nbytes(tvq));
    ggml_backend_tensor_set(tkd,  kd.data(),  0, ggml_nbytes(tkd));
    ggml_backend_tensor_set(tvd,  vd.data(),  0, ggml_nbytes(tvd));
    ggml_backend_tensor_set(tidx, idx.data(), 0, ggml_nbytes(tidx));
    ggml_backend_tensor_set(tpos, pos.data(), 0, ggml_nbytes(tpos));
    ggml_backend_graph_compute(backend, gf);

    auto get = [](ggml_tensor * t) {
        std::vector<float> x(ggml_nelements(t));
        ggml_backend_tensor_get(t, x.data(), 0, ggml_nbytes(t));
        return x;
    };
    quant_result r;
    r.max_abs = std::max(max_abs_diff(get(oq), get(od), &r.nmse), max_abs_diff(get(oqw), get(odw), &r.nmse));
    r.cast_max_abs = std::max(max_abs_diff(get(kc), kd), max_abs_diff(get(vc), vd));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 512;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int NB = N / BS;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t) D * Hq * T), kcmp((size_t) D * Hkv * NB);
    std::vector<ggml_fp16_t> k16((size_t) D * Hkv * N), v16((size_t) D * Hkv * N);
    for (auto & x : q) x = nd(rng);
    for (auto & x : kcmp) x = 0.5f * nd(rng);
    for (auto & x : k16) x = ggml_fp32_to_fp16(nd(rng));
    for (auto & x : v16) x = ggml_fp32_to_fp16(nd(rng));
    // F32 K/V to be quantized (check 3)
    std::vector<float> k32(k16.size()), v32(v16.size());
    for (auto & x : k32) x = nd(rng);
    for (auto & x : v32) x = nd(rng);

    // queries: a few early positions (no complete block / fewer than S candidates) and the rest at the end of the context
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = (t < 8 && T > 8) ? t * 97 : N - T + t;

    // valid block indices: cur always, plus distinct random blocks <= cur, padded with -1
    std::vector<int32_t> idx((size_t) S * Hkv * T, -1);
    for (int t = 0; t < T; ++t) {
        const int cur = pos[t] / BS;
        for (int h = 0; h < Hkv; ++h) {
            std::vector<int> cand;
            for (int b = 0; b <= cur; ++b) if (b != cur) cand.push_back(b);
            std::shuffle(cand.begin(), cand.end(), rng);
            int32_t * row = &idx[((size_t) t * Hkv + h) * S];
            row[0] = cur;
            const int n = std::min<int>((int) cand.size(), S - 1 - (int) (rng() % 3));
            for (int s = 0; s < n; ++s) row[1 + s] = cand[s];
        }
    }

    ggml_backend_load_all();
    int fails = 0;
    std::vector<std::tuple<bool, std::string, std::vector<int32_t>>> sels; // (is cpu, name, select output)
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, 32);

        run_result r = run(be, T, N, q, k16, v16, idx, pos, kcmp);

        // 1.-3. sel_attn (V rows and transposed V) vs dense masked attention, sparse window vs dense sliding window
        double e_slc, d_slc, e_tr, d_tr, e_swa, d_swa, e_rt, d_rt;
        nmse_of(r.o_op,     r.o_ref,     e_slc, d_slc);
        nmse_of(r.o_op_t,   r.o_ref,     e_tr,  d_tr);
        nmse_of(r.o_swa_op, r.o_swa_ref, e_swa, d_swa);
        nmse_of(r.o_op_t,   r.o_op,      e_rt,  d_rt);

        // 4. select vs mul_mat/soft_max/mean/top_k, compared as sets. ggml_top_k does not order its output, so the
        // reference set is the candidates among all its S entries (every candidate ranks above the -inf non-candidates).
        // The effective score is 1 for a forced block and the reference softmax score for the other complete blocks.
        int rows = 0, exact = 0, bad = 0; double worst_gap = 0;
        for (int t = 0; t < T; ++t) {
            const int cur = pos[t] / BS;
            if ((cur + 1) * BS - 1 > pos[t] && cur < 1) continue; // no complete block: reference softmax is NaN
            auto forced    = [&](int b) { return b <= cur && (b == 0 || b == cur || b == cur - 1); };
            auto candidate = [&](int b) { return b >= 0 && b < NB && ((b + 1) * BS - 1 <= pos[t] || forced(b)); };
            int n_cand = 0;
            for (int b = 0; b < NB; ++b) n_cand += candidate(b);
            for (int h = 0; h < Hkv; ++h) {
                const int32_t * a = &r.sel_op[((size_t) t * Hkv + h) * S];
                const int32_t * b = &r.sel_ref[((size_t) t * Hkv + h) * S];
                const float * sc = &r.score_ref[((size_t) h * T + t) * NB];
                auto eff = [&](int x) { return forced(x) ? 1.0 : (double) sc[x]; };
                std::set<int> sa, sb;
                bool valid = true;
                for (int s = 0; s < S; ++s) {
                    if (a[s] < 0) continue;
                    valid = valid && candidate(a[s]) && !sa.count(a[s]);
                    sa.insert(a[s]);
                }
                for (int s = 0; s < S; ++s) if (candidate(b[s])) sb.insert(b[s]);
                rows++;
                if (valid && sa == sb) { exact++; continue; }
                // score gap of the swapped blocks (near-tie check)
                double gap = 0;
                for (int x : sa) if (!sb.count(x)) for (int y : sb) if (!sa.count(y)) gap = std::max(gap, std::fabs(eff(x) - eff(y)));
                worst_gap = std::max(worst_gap, gap);
                // only near-tie swaps between valid candidates are acceptable
                bad += !valid || gap > 1e-5 || sa.size() != sb.size() || (int) sa.size() != std::min(S, n_cand);
            }
        }
        const bool ok = e_slc < 1e-6 && e_tr < 1e-6 && e_swa < 1e-6 && bad == 0;
        fails += !ok;
        printf("%-8s sel_attn vs dense-mask: nmse %.2e max_abs %.2e | transposed V: nmse %.2e (vs V rows: max_abs %.2e) | "
               "sparse window vs dense window: nmse %.2e max_abs %.2e | select vs topk-composition: %d/%d sets equal, "
               "worst swapped score gap %.2e  %s\n",
               ggml_backend_name(be), e_slc, d_slc, e_tr, d_rt, e_swa, d_swa, exact, rows, worst_gap, ok ? "OK" : "FAIL");
        sels.emplace_back(ggml_backend_is_cpu(be), ggml_backend_name(be), r.sel_op);

        // quantized K/V: same op on the dequantized values must agree to float rounding (in practice bit-exactly)
        for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
            const quant_result qr = run_quant(be, type, T, N, q, k32, v32, idx, pos);
            const bool qok = qr.nmse < 1e-10 && qr.cast_max_abs == 0.0;
            fails += !qok;
            printf("%-8s %s K/V: sel_attn(quant) vs sel_attn(dequant F32): nmse %.2e max_abs %.2e | cast vs host dequant: max_abs %.2e  %s\n",
                   ggml_backend_name(be), ggml_type_name(type), qr.nmse, qr.max_abs, qr.cast_max_abs, qok ? "OK" : "FAIL");
        }
        ggml_backend_free(be);
    }
    // 4. select is computed with the same float operations on every backend (ggml-nsa-exp.h), so it must match the
    // CPU output bit for bit (and not only up to near-tie swaps)
    for (const auto & [is_cpu_ref, name_ref, sel_ref] : sels) {
        if (!is_cpu_ref) continue;
        for (const auto & [is_cpu, name, sel] : sels) {
            if (is_cpu) continue;
            int same = 0, n = 0;
            for (size_t r0 = 0; r0 < sel.size(); r0 += S, ++n) {
                same += memcmp(&sel[r0], &sel_ref[r0], S * sizeof(int32_t)) == 0;
            }
            fails += same != n;
            printf("%-8s select vs %s: %d/%d rows identical  %s\n", name.c_str(), name_ref.c_str(), same, n, same == n ? "OK" : "FAIL");
        }
    }
    return fails ? 1 : 0;
}
