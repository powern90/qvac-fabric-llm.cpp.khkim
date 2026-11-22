// Level-1 validation of the Gated DeltaNet operator used by the gdn-nsa GDN layers.
//
// The model runs ggml_gated_delta_net(q, k, v, g, beta, s0, K=1) for both prefill (T > 1) and
// decode (T = 1): build_layer_gdn -> build_recurrent_attn -> build_delta_net -> build_delta_net_fused
// (fused_gdn_ar / fused_gdn_ch are hard-set true, n_rs_seq is clamped to 0 for this arch).
//
// Op contract (ggml/include/ggml.h, CPU ggml/src/ggml-cpu/ops.cpp), per sequence and head,
// with S[i][j] (i = key dim, j = value dim) stored at state[(seq,h) base + j*D + i]:
//   S   <- exp(g_t) * S
//   u_j  = beta_t * (v_t[j] - sum_i S[i][j] k_t[i])
//   S[i][j] += k_t[i] * u_j
//   o_t[j]  = (1/sqrt(D)) * sum_i S[i][j] q_t[i]
// output = [o (D*H*T*n_seqs floats) | final state (D*D*H*n_seqs floats)].
// This is the fla recurrence h[K,V]: h = exp(g) h; h += k (beta (v - h^T k))^T; o = h^T (q * K^-0.5),
// with the ggml state memory equal to h transposed (ne0 = key dim). The oracle below is written in the
// fla h[i][j] form in fp64 and mapped to the ggml layout on comparison, so orientation is checked too.
//
// Paths:
//   A: one op call over all T tokens (prefill)
//   B: T chained single-token op calls, state carried through the op output (decode)
//
// usage: gdn-oracle-test [json_path] (default l1_gdn.json)
// exit code 1 on any failure.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// thresholds (see report / JSON "thresholds")
static const double OUT_NRMSE_TOL   = 1e-4; // output: relative (NRMSE)
static const double STATE_ATOL      = 1e-5; // state: absolute max error
static const double ZERO_REF_ATOL   = 1e-6; // used when the reference norm is ~0

struct inputs {
    int H, D, T, NS;
    std::vector<float> q, k, v, g, b, s0;
};

struct cfg {
    std::string name;
    int H, D, T, NS;
    bool zero_s0    = false;
    bool zero_key   = false;
    bool zero_value = false;
    float beta_fix  = -1.0f; // < 0: random sigmoid
    bool g_fix      = false;
    float g_val     = 0.0f;
    uint32_t seed   = 1;
};

static inputs make_inputs(const cfg & c) {
    inputs in;
    in.H = c.H; in.D = c.D; in.T = c.T; in.NS = c.NS;
    const int H = c.H, D = c.D, T = c.T, NS = c.NS;
    std::mt19937 rng(c.seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    in.q.resize((size_t) D * H * T * NS);
    in.k.resize(in.q.size());
    in.v.resize(in.q.size());
    in.g.resize((size_t) H * T * NS);
    in.b.resize(in.g.size());
    in.s0.resize((size_t) D * D * H * NS);
    // q, k: L2-normalized as in build_gdn_l2_norm (x / sqrt(max(sum x^2, eps^2))-like; eps 1e-6)
    auto l2 = [&](std::vector<float> & x, bool zero) {
        for (size_t r = 0; r < x.size() / D; ++r) {
            std::vector<double> t(D);
            double ss = 0;
            for (int i = 0; i < D; ++i) { t[i] = nd(rng); ss += t[i] * t[i]; }
            const double inv = 1.0 / std::max(std::sqrt(ss), 1e-6);
            for (int i = 0; i < D; ++i) x[r * D + i] = zero ? 0.0f : (float) (t[i] * inv);
        }
    };
    l2(in.q, false);
    l2(in.k, c.zero_key);
    for (auto & x : in.v) x = c.zero_value ? 0.0f : (float) nd(rng);
    // g = -exp(A_log) * softplus(a + dt_bias) <= 0; realistic range [-3, -1e-4], log-uniform
    for (auto & x : in.g) {
        x = c.g_fix ? c.g_val : (float) -std::exp(std::log(1e-4) + ud(rng) * (std::log(3.0) - std::log(1e-4)));
    }
    for (auto & x : in.b) x = c.beta_fix >= 0 ? c.beta_fix : (float) (1.0 / (1.0 + std::exp(-2.0 * nd(rng))));
    for (auto & x : in.s0) x = c.zero_s0 ? 0.0f : (float) (0.1 * nd(rng));
    return in;
}

// fp64 oracle in fla form. returns o in ggml layout and the final state in ggml layout.
// variant != 0 deliberately breaks the definition (negative controls):
//   1 = state orientation transposed (mem[i*D+j] = S[i][j]), 2 = no q scale,
//   3 = delta prediction uses the undecayed state (S = a S + beta (v - S k) k^T)
static void oracle(const inputs & in, std::vector<double> & o, std::vector<double> & st, int variant = 0) {
    const int H = in.H, D = in.D, T = in.T, NS = in.NS;
    const double scale = variant == 2 ? 1.0 : 1.0 / std::sqrt((double) D);
    auto mi = [&](int i, int j) { return variant == 1 ? (size_t) i * D + j : (size_t) j * D + i; };
    o.assign((size_t) D * H * T * NS, 0.0);
    st.assign((size_t) D * D * H * NS, 0.0);
    std::vector<double> h((size_t) D * D), u(D);
    for (int s = 0; s < NS; ++s) {
        for (int hh = 0; hh < H; ++hh) {
            const size_t sb = ((size_t) s * H + hh) * D * D;
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j) h[(size_t) i * D + j] = in.s0[sb + mi(i, j)]; // h = S0
            for (int t = 0; t < T; ++t) {
                const size_t rb = (((size_t) s * T + t) * H + hh) * D;
                const float * q = &in.q[rb];
                const float * k = &in.k[rb];
                const float * v = &in.v[rb];
                const double alpha = std::exp((double) in.g[((size_t) s * T + t) * H + hh]);
                const double beta  = in.b[((size_t) s * T + t) * H + hh];
                if (variant != 3) for (auto & x : h) x *= alpha;
                for (int j = 0; j < D; ++j) {
                    double kv = 0;
                    for (int i = 0; i < D; ++i) kv += h[(size_t) i * D + j] * (double) k[i];
                    u[j] = beta * ((double) v[j] - kv);
                }
                if (variant == 3) for (auto & x : h) x *= alpha;
                for (int i = 0; i < D; ++i)
                    for (int j = 0; j < D; ++j) h[(size_t) i * D + j] += (double) k[i] * u[j];
                for (int j = 0; j < D; ++j) {
                    double acc = 0;
                    for (int i = 0; i < D; ++i) acc += h[(size_t) i * D + j] * ((double) q[i] * scale);
                    o[rb + j] = acc;
                }
            }
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j) st[sb + mi(i, j)] = h[(size_t) i * D + j];
        }
    }
}

struct op_result {
    std::vector<float> o, st;
    bool ok = false;
};

// path A: one call over all T tokens
static op_result run_A(ggml_backend_t be, const inputs & in) {
    const int H = in.H, D = in.D, T = in.T, NS = in.NS;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, NS);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, NS);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, NS);
    ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, NS);
    ggml_tensor * b  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, NS);
    ggml_tensor * s0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, NS);
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s0, 1);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    op_result r;
    if (!ggml_backend_supports_op(be, out)) { ggml_free(ctx); return r; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(g, in.g.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, in.b.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(s0, in.s0.data(), 0, ggml_nbytes(s0));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> all(ggml_nelements(out));
    ggml_backend_tensor_get(out, all.data(), 0, ggml_nbytes(out));
    const size_t no = (size_t) D * H * T * NS, ns = (size_t) D * D * H * NS;
    r.o.assign(all.begin(), all.begin() + no);
    r.st.assign(all.begin() + no, all.begin() + no + ns);
    r.ok = true;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// path B: T chained single-token calls; the state input of call t is the state tail of call t-1
// (the same view build_delta_net_fused takes as new_state)
static op_result run_B(ggml_backend_t be, const inputs & in) {
    const int H = in.H, D = in.D, T = in.T, NS = in.NS;
    const size_t n_tensors = (size_t) T * 10 + 64;
    ggml_init_params ip = { n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom((size_t) T * 8 + 64, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * s0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, NS);
    std::vector<ggml_tensor *> tq(T), tk(T), tv(T), tg(T), tb(T), tout(T);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t) T * 8 + 64, false);
    ggml_tensor * st = s0;
    op_result r;
    for (int t = 0; t < T; ++t) {
        tq[t] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, 1, NS);
        tk[t] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, 1, NS);
        tv[t] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, 1, NS);
        tg[t] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, NS);
        tb[t] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, NS);
        tout[t] = ggml_gated_delta_net(ctx, tq[t], tk[t], tv[t], tg[t], tb[t], st, 1);
        ggml_set_output(tout[t]);
        if (t == 0 && !ggml_backend_supports_op(be, tout[t])) { ggml_free(ctx); return r; }
        st = ggml_view_4d(ctx, tout[t], D, D, H, NS,
                ggml_row_size(GGML_TYPE_F32, D), ggml_row_size(GGML_TYPE_F32, (size_t) D * D),
                ggml_row_size(GGML_TYPE_F32, (size_t) D * D * H), ggml_row_size(GGML_TYPE_F32, (size_t) D * H * NS));
        ggml_build_forward_expand(gf, tout[t]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(s0, in.s0.data(), 0, ggml_nbytes(s0));
    std::vector<float> tmp;
    auto slice = [&](const std::vector<float> & src, int t, int row) {
        tmp.resize((size_t) row * H * NS);
        for (int s = 0; s < NS; ++s)
            memcpy(&tmp[(size_t) s * H * row], &src[(((size_t) s * T + t) * H) * row], sizeof(float) * H * row);
        return tmp.data();
    };
    for (int t = 0; t < T; ++t) {
        ggml_backend_tensor_set(tq[t], slice(in.q, t, D), 0, ggml_nbytes(tq[t]));
        ggml_backend_tensor_set(tk[t], slice(in.k, t, D), 0, ggml_nbytes(tk[t]));
        ggml_backend_tensor_set(tv[t], slice(in.v, t, D), 0, ggml_nbytes(tv[t]));
        ggml_backend_tensor_set(tg[t], slice(in.g, t, 1), 0, ggml_nbytes(tg[t]));
        ggml_backend_tensor_set(tb[t], slice(in.b, t, 1), 0, ggml_nbytes(tb[t]));
    }
    ggml_backend_graph_compute(be, gf);
    r.o.resize((size_t) D * H * T * NS);
    std::vector<float> all;
    for (int t = 0; t < T; ++t) {
        all.resize(ggml_nelements(tout[t]));
        ggml_backend_tensor_get(tout[t], all.data(), 0, ggml_nbytes(tout[t]));
        for (int s = 0; s < NS; ++s)
            memcpy(&r.o[(((size_t) s * T + t) * H) * D], &all[(size_t) s * H * D], sizeof(float) * H * D);
        if (t == T - 1) {
            const size_t off = (size_t) D * H * NS;
            r.st.assign(all.begin() + off, all.begin() + off + (size_t) D * D * H * NS);
        }
    }
    r.ok = true;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

struct metrics {
    double max_abs = 0, rmse = 0, nrmse = 0, cos = 0, ref_norm = 0;
    long nonfinite = 0;
};

template <typename R>
static metrics compare(const std::vector<R> & ref, const std::vector<float> & x) {
    metrics m;
    double dd = 0, rr = 0, xx = 0, rx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double xv = x[i];
        if (!std::isfinite(xv)) { m.nonfinite++; continue; }
        const double rv = ref[i];
        const double d = xv - rv;
        dd += d * d; rr += rv * rv; xx += xv * xv; rx += rv * xv;
        m.max_abs = std::max(m.max_abs, std::fabs(d));
    }
    m.rmse = std::sqrt(dd / std::max<size_t>(ref.size(), 1));
    m.ref_norm = std::sqrt(rr);
    m.nrmse = m.ref_norm > 1e-30 ? std::sqrt(dd) / m.ref_norm : (dd == 0 ? 0.0 : INFINITY);
    m.cos = (rr > 0 && xx > 0) ? rx / std::sqrt(rr * xx) : (rr == 0 && xx == 0 ? 1.0 : 0.0);
    return m;
}

static bool pass_out(const metrics & m) {
    if (m.nonfinite) return false;
    if (m.ref_norm < 1e-12) return m.max_abs <= ZERO_REF_ATOL;
    return m.nrmse <= OUT_NRMSE_TOL;
}
static bool pass_state(const metrics & m) {
    if (m.nonfinite) return false;
    return m.max_abs <= STATE_ATOL;
}

struct record {
    std::string group, cname, backend, cmp, tensor;
    int H, D, T, NS;
    metrics m;
    bool pass;
};

static std::vector<record> g_recs;
static int g_fail = 0;

static void add(const std::string & group, const cfg & c, const char * be, const char * cmp, const char * tensor,
                const metrics & m, bool pass) {
    g_recs.push_back({group, c.name, be, cmp, tensor, c.H, c.D, c.T, c.NS, m, pass});
    if (!pass) g_fail++;
    printf("%-6s %-16s %-6s H=%d D=%03d T=%03d ns=%d %-12s %-5s max_abs=%.3e rmse=%.3e nrmse=%.3e cos=%.9f nonfin=%ld %s\n",
           group.c_str(), c.name.c_str(), be, c.H, c.D, c.T, c.NS, cmp, tensor, m.max_abs, m.rmse, m.nrmse, m.cos,
           m.nonfinite, pass ? "OK" : "FAIL");
}

static void run_case(ggml_backend_t be, const char * bename, const std::string & group, const cfg & c, bool do_B) {
    inputs in = make_inputs(c);
    std::vector<double> o_ref, s_ref;
    oracle(in, o_ref, s_ref);
    op_result A = run_A(be, in);
    if (!A.ok) {
        printf("%-6s %-16s %-6s SKIP (op not supported)\n", group.c_str(), c.name.c_str(), bename);
        return;
    }
    {
        metrics mo = compare(o_ref, A.o), ms = compare(s_ref, A.st);
        add(group, c, bename, "A_vs_oracle", "out", mo, pass_out(mo));
        add(group, c, bename, "A_vs_oracle", "state", ms, pass_state(ms));
    }
    if (do_B) {
        op_result B = run_B(be, in);
        if (!B.ok) {
            printf("%-6s %-16s %-6s B SKIP (op not supported)\n", group.c_str(), c.name.c_str(), bename);
            return;
        }
        metrics mo = compare(o_ref, B.o), ms = compare(s_ref, B.st);
        add(group, c, bename, "B_vs_oracle", "out", mo, pass_out(mo));
        add(group, c, bename, "B_vs_oracle", "state", ms, pass_state(ms));
        metrics ao = compare(A.o, B.o), as = compare(A.st, B.st);
        add(group, c, bename, "A_vs_B", "out", ao, pass_out(ao));
        add(group, c, bename, "A_vs_B", "state", as, pass_state(as));
    }
}

// oracle self-checks against closed forms (independent of the op)
static void oracle_selfcheck() {
    // no write (beta = 0 or k = 0): S_T = exp(sum g) S0, o_t = scale * exp(sum_{<=t} g) S0^T q_t
    for (int mode = 0; mode < 2; ++mode) {
        cfg c{mode == 0 ? "self_beta0" : "self_zero_key", 2, 16, 33, 1};
        if (mode == 0) c.beta_fix = 0.0f; else c.zero_key = true;
        c.seed = 77;
        inputs in = make_inputs(c);
        std::vector<double> o, st;
        oracle(in, o, st);
        double err = 0;
        for (int h = 0; h < in.H; ++h) {
            double G = 0;
            for (int t = 0; t < in.T; ++t) G += in.g[(size_t) t * in.H + h];
            for (size_t e = 0; e < (size_t) in.D * in.D; ++e) {
                const size_t idx = (size_t) h * in.D * in.D + e;
                err = std::max(err, std::fabs(st[idx] - std::exp(G) * in.s0[idx]));
            }
        }
        const bool ok = err < 1e-12;
        printf("oracle self-check %-14s closed-form state max_abs=%.3e %s\n", c.name.c_str(), err, ok ? "OK" : "FAIL");
        if (!ok) g_fail++;
    }
}

static void write_json(const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); g_fail++; return; }
    fprintf(f, "{\n  \"op\": \"ggml_gated_delta_net(q,k,v,g,beta,state,K=1)\",\n");
    fprintf(f, "  \"definition\": \"S=exp(g)S; u=beta(v-S^T k); S+=k u^T; o=(1/sqrt(D)) S^T q; ggml state mem[j*D+i]=S[i][j], i=key dim\",\n");
    fprintf(f, "  \"thresholds\": {\"out_nrmse\": %g, \"state_max_abs\": %g, \"zero_ref_max_abs\": %g},\n",
            OUT_NRMSE_TOL, STATE_ATOL, ZERO_REF_ATOL);
    fprintf(f, "  \"n_fail\": %d,\n  \"records\": [\n", g_fail);
    for (size_t i = 0; i < g_recs.size(); ++i) {
        const record & r = g_recs[i];
        auto num = [](double x) { char b[64]; if (std::isfinite(x)) snprintf(b, sizeof b, "%.6e", x); else snprintf(b, sizeof b, "null"); return std::string(b); };
        fprintf(f, "    {\"group\": \"%s\", \"case\": \"%s\", \"backend\": \"%s\", \"H\": %d, \"D\": %d, \"T\": %d, \"n_seqs\": %d, "
                   "\"compare\": \"%s\", \"tensor\": \"%s\", \"max_abs\": %s, \"rmse\": %s, \"nrmse\": %s, \"cos\": %.12f, "
                   "\"ref_norm\": %s, \"nonfinite\": %ld, \"pass\": %s}%s\n",
                r.group.c_str(), r.cname.c_str(), r.backend.c_str(), r.H, r.D, r.T, r.NS, r.cmp.c_str(), r.tensor.c_str(),
                num(r.m.max_abs).c_str(), num(r.m.rmse).c_str(), num(r.m.nrmse).c_str(), r.m.cos, num(r.m.ref_norm).c_str(),
                r.m.nonfinite, r.pass ? "true" : "false", i + 1 < g_recs.size() ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

int main(int argc, char ** argv) {
    const char * json_path = argc > 1 ? argv[1] : "l1_gdn.json";

    oracle_selfcheck();

    const int Ts[] = { 1, 2, 8, 63, 64, 65, 128, 129, 512 };
    struct shape { int H, D; };
    const shape shapes[] = { { 8, 128 }, { 2, 16 } };

    ggml_backend_load_all();
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, 16);
        const std::string bn = ggml_backend_is_cpu(be) ? "CPU" : ggml_backend_name(be);
        printf("== backend %s (%s)\n", bn.c_str(), ggml_backend_dev_description(dev));

        // P0: op vs oracle over T, plus chunk (A) vs token-by-token (B)
        for (const shape & sh : shapes) {
            for (int T : Ts) {
                cfg c{"base_s0rand", sh.H, sh.D, T, 1};
                c.seed = 1000 + T * 7 + sh.D;
                run_case(be, bn.c_str(), "P0", c, /*do_B=*/true);
            }
        }
        // multi-sequence (state seq stride)
        {
            cfg c{"base_ns2", 8, 128, 65, 2};
            c.seed = 4242;
            run_case(be, bn.c_str(), "P0", c, true);
        }

        // negative controls: broken oracles must be rejected by the same metrics
        {
            cfg c{"negctl", 8, 128, 65, 1};
            c.seed = 31337;
            inputs in = make_inputs(c);
            op_result A = run_A(be, in);
            const char * names[] = { "", "transposed_state", "no_q_scale", "decay_after_pred" };
            for (int var = 1; var <= 3; ++var) {
                std::vector<double> o, st;
                oracle(in, o, st, var);
                metrics mo = compare(o, A.o), ms = compare(st, A.st);
                const bool detected = !pass_out(mo) || !pass_state(ms);
                printf("NEGCTL %-16s %-6s out nrmse=%.3e state max_abs=%.3e -> %s\n", names[var], bn.c_str(), mo.nrmse,
                       ms.max_abs, detected ? "detected (OK)" : "NOT DETECTED (FAIL)");
                if (!detected) g_fail++;
                cfg cc = c; cc.name = std::string("negctl_") + names[var];
                g_recs.push_back({"NEG", cc.name, bn, "A_vs_broken", "out", c.H, c.D, c.T, c.NS, mo, detected});
                g_recs.push_back({"NEG", cc.name, bn, "A_vs_broken", "state", c.H, c.D, c.T, c.NS, ms, detected});
            }
        }

        // P1: edge cases (model shape, T = 65, 129, 512)
        for (int T : { 65, 129, 512 }) {
            std::vector<cfg> ec;
            { cfg c{"s0_zero", 8, 128, T, 1};      c.zero_s0 = true;             ec.push_back(c); }
            { cfg c{"s0_nonzero", 8, 128, T, 1};                                 ec.push_back(c); }
            { cfg c{"zero_key", 8, 128, T, 1};     c.zero_key = true;            ec.push_back(c); }
            { cfg c{"zero_value", 8, 128, T, 1};   c.zero_value = true;          ec.push_back(c); }
            { cfg c{"zero_value_s0z", 8, 128, T, 1}; c.zero_value = true; c.zero_s0 = true; ec.push_back(c); }
            { cfg c{"beta0", 8, 128, T, 1};        c.beta_fix = 0.0f;            ec.push_back(c); }
            { cfg c{"beta1", 8, 128, T, 1};        c.beta_fix = 1.0f;            ec.push_back(c); }
            { cfg c{"g0", 8, 128, T, 1};           c.g_fix = true; c.g_val = 0.0f;    ec.push_back(c); }
            { cfg c{"g0_beta1", 8, 128, T, 1};     c.g_fix = true; c.g_val = 0.0f; c.beta_fix = 1.0f; ec.push_back(c); }
            { cfg c{"g_-60", 8, 128, T, 1};        c.g_fix = true; c.g_val = -60.0f;  ec.push_back(c); }
            { cfg c{"g_-1000", 8, 128, T, 1};      c.g_fix = true; c.g_val = -1000.0f; ec.push_back(c); }
            uint32_t seed = 9000 + T;
            for (cfg & c : ec) { c.seed = seed++; run_case(be, bn.c_str(), "P1", c, true); }
        }
        ggml_backend_free(be);
    }

    // error growth vs T (P0 base, out NRMSE / state max_abs)
    printf("\n== error growth vs T (case base_s0rand)\n");
    printf("%-6s %-4s %-12s | %-9s", "backend", "D", "compare", "tensor");
    for (int T : Ts) printf(" T=%-8d", T);
    printf("\n");
    std::vector<std::string> bes;
    for (const record & r : g_recs) if (std::find(bes.begin(), bes.end(), r.backend) == bes.end()) bes.push_back(r.backend);
    for (const std::string & b : bes) for (const shape & sh : shapes) for (const char * cmp : { "A_vs_oracle", "B_vs_oracle", "A_vs_B" })
    for (const char * tensor : { "out", "state" }) {
        printf("%-7s %-4d %-12s | %-5s %s", b.c_str(), sh.D, cmp, tensor, tensor[0] == 'o' ? "nrmse" : "maxab");
        for (int T : Ts) {
            double val = NAN;
            for (const record & r : g_recs)
                if (r.backend == b && r.cname == "base_s0rand" && r.D == sh.D && r.T == T && r.cmp == cmp && r.tensor == tensor)
                    val = tensor[0] == 'o' ? r.m.nrmse : r.m.max_abs;
            printf(" %-10.3e", val);
        }
        printf("\n");
    }

    write_json(json_path);
    printf("\n%zu checks, %d failed; json: %s\n", g_recs.size(), g_fail, json_path);
    return g_fail ? 1 : 0;
}
