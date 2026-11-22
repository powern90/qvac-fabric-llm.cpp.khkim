// Level 3 / level 4 validation of the gdn-nsa port (KV cache + recurrent state correctness).
//
//   L3 "lengths": prefill only the first L tokens in one ubatch, for L at the NSA block (64) and window (512)
//                 boundaries. Compared with the full-sequence prefill of the same backend (causality: row t must not
//                 depend on the sequence length) and with the PyTorch reference rows [0, L).
//   L4 "splits":  prefill the first S tokens in one ubatch, then decode the rest token by token. Compared with the
//                 full-sequence prefill and with the reference. S at 63/64/65 makes the decode step complete a block,
//                 S at 511/512/513 makes it cross the sliding-window boundary, S = 1 is pure decode.
//
// usage: nsa-cache-equiv -m model.gguf -t tokens.txt [-ngl N] [--ref logits.npy] [--ref32 logits_fp32.npy]
//                        [--splits 1,63,64] [--lengths 1,2,63] [--json out.json] [-ctk f16] [-ctv f16]
// metrics per run (rows that the run computed): max |diff|, NRMSE = ||a-b|| / ||b||, min per-row cosine, mean/max
// KL(ref || ours), top-1 agreement, and the number of non-finite logits.
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<float> load_npy_f32(const std::string & path, long & rows, long & cols) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    char magic[6];
    unsigned char ver[2];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0 || fread(ver, 1, 2, f) != 2) {
        fprintf(stderr, "%s: not a .npy file\n", path.c_str()); exit(1);
    }
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h; if (fread(&h, 2, 1, f) != 1) exit(1); hlen = h; }
    else             { if (fread(&hlen, 4, 1, f) != 1) exit(1); }
    std::string hdr(hlen, ' ');
    if (fread(&hdr[0], 1, hlen, f) != hlen) exit(1);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos) {
        fprintf(stderr, "%s: need little-endian float32 C-order, header %s\n", path.c_str(), hdr.c_str()); exit(1);
    }
    const size_t sp = hdr.find("'shape': (");
    if (sp == std::string::npos || sscanf(hdr.c_str() + sp + 10, "%ld, %ld", &rows, &cols) != 2) {
        fprintf(stderr, "%s: cannot parse 2-d shape from %s\n", path.c_str(), hdr.c_str()); exit(1);
    }
    std::vector<float> v((size_t) rows * cols);
    if (fread(v.data(), sizeof(float), v.size(), f) != v.size()) { fprintf(stderr, "%s: short read\n", path.c_str()); exit(1); }
    fclose(f);
    return v;
}

static std::vector<int> parse_list(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) out.push_back(std::stoi(x));
    return out;
}

struct metrics {
    double max_abs = 0, nrmse = 0, cos_min = 1, kl_mean = 0, kl_max = 0, top1 = 0;
    long   nonfinite = 0, rows = 0;
};

// ours, ref: [rows][V] starting at row r0 of both buffers
static metrics compare(const float * ours, const float * ref, long r0, long r1, long V) {
    metrics m;
    double sd = 0, sr = 0, kls = 0;
    long agree = 0;
    std::vector<double> lp(V), lq(V);
    for (long r = r0; r < r1; ++r) {
        const float * a = ours + (size_t) r * V;
        const float * b = ref  + (size_t) r * V;
        double dd = 0, aa = 0, bb = 0, ab = 0, ma = -INFINITY, mb = -INFINITY;
        long ia = 0, ib = 0;
        for (long j = 0; j < V; ++j) {
            if (!std::isfinite(a[j])) { m.nonfinite++; continue; }
            const double d = (double) a[j] - b[j];
            m.max_abs = std::max(m.max_abs, std::fabs(d));
            dd += d*d; aa += (double) a[j]*a[j]; bb += (double) b[j]*b[j]; ab += (double) a[j]*b[j];
            if (a[j] > ma) { ma = a[j]; ia = j; }
            if (b[j] > mb) { mb = b[j]; ib = j; }
        }
        sd += dd; sr += bb;
        m.cos_min = std::min(m.cos_min, ab / std::sqrt(std::max(aa*bb, 1e-300)));
        agree += ia == ib;
        double za = 0, zb = 0;
        for (long j = 0; j < V; ++j) { za += std::exp((double) a[j] - ma); zb += std::exp((double) b[j] - mb); }
        za = std::log(za); zb = std::log(zb);
        double kl = 0;
        for (long j = 0; j < V; ++j) {
            const double lb = (double) b[j] - mb - zb; // ref
            const double la = (double) a[j] - ma - za; // ours
            kl += std::exp(lb) * (lb - la);
        }
        kls += kl;
        m.kl_max = std::max(m.kl_max, kl);
    }
    m.rows    = r1 - r0;
    m.nrmse   = std::sqrt(sd / std::max(sr, 1e-300));
    m.kl_mean = kls / std::max<long>(1, m.rows);
    m.top1    = (double) agree / std::max<long>(1, m.rows);
    return m;
}

static std::string mjson(const metrics & m) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"rows\": %ld, \"max_abs\": %.4e, \"nrmse\": %.4e, \"cos_min\": %.8f, \"kl_mean\": %.4e, \"kl_max\": %.4e, "
             "\"top1\": %.6f, \"nonfinite\": %ld}",
             m.rows, m.max_abs, m.nrmse, m.cos_min, m.kl_mean, m.kl_max, m.top1, m.nonfinite);
    return buf;
}

int main(int argc, char ** argv) {
    std::string model_path, tok_path, ref_path, ref32_path, json_path, ctk = "f16", ctv = "f16";
    std::string splits_s = "1,63,64,65,127,128,511,512,513", lengths_s = "1,2,63,64,65,127,128,129,511,512,513";
    int ngl = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "-m")        model_path = next();
        else if (a == "-t")        tok_path   = next();
        else if (a == "-ngl")      ngl        = std::stoi(next());
        else if (a == "--ref")     ref_path   = next();
        else if (a == "--ref32")   ref32_path = next();
        else if (a == "--splits")  splits_s   = next();
        else if (a == "--lengths") lengths_s  = next();
        else if (a == "--json")    json_path  = next();
        else if (a == "-ctk")      ctk        = next();
        else if (a == "-ctv")      ctv        = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    std::vector<llama_token> toks;
    {
        std::ifstream f(tok_path);
        long v;
        while (f >> v) toks.push_back((llama_token) v);
    }
    const int T = (int) toks.size();

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    auto kv_type = [](const std::string & s) {
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            const char * name = ggml_type_name((ggml_type) t);
            if (name && s == name) return (ggml_type) t;
        }
        fprintf(stderr, "unknown cache type %s\n", s.c_str());
        exit(1);
    };
    auto cparams = llama_context_default_params();
    cparams.n_ctx     = std::max(1024, (T + 255) / 256 * 256);
    cparams.n_batch   = T;
    cparams.n_ubatch  = T; // every prefill below is a single ubatch
    cparams.n_seq_max = 1;
    cparams.type_k    = kv_type(ctk);
    cparams.type_v    = kv_type(ctv);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const long V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(T, 0, 1);

    // prefill [0, S) as one ubatch, then decode [S, n_end) token by token; logits of every row in out
    auto run = [&](int S, int n_end, std::vector<float> & out) {
        llama_memory_clear(llama_get_memory(ctx), true);
        out.assign((size_t) n_end * V, NAN);
        int p0 = 0;
        while (p0 < n_end) {
            const int n = p0 == 0 ? S : 1;
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = toks[p0 + i];
                batch.pos[i] = p0 + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d (S = %d)\n", p0, S); exit(1); }
            for (int i = 0; i < n; ++i) {
                memcpy(&out[(size_t) (p0 + i) * V], llama_get_logits_ith(ctx, i), sizeof(float) * V);
            }
            p0 += n;
        }
    };

    std::vector<float> ref, ref32;
    long rr = 0, rc = 0;
    if (!ref_path.empty()) {
        ref = load_npy_f32(ref_path, rr, rc);
        if (rr < T || rc != V) { fprintf(stderr, "ref shape [%ld, %ld] does not fit T = %d, V = %ld\n", rr, rc, T, V); return 1; }
    }
    if (!ref32_path.empty()) {
        ref32 = load_npy_f32(ref32_path, rr, rc);
        if (rr < T || rc != V) { fprintf(stderr, "ref32 shape [%ld, %ld] does not fit\n", rr, rc); return 1; }
    }

    std::vector<float> full, cur;
    run(T, T, full);

    std::string js = "{\n \"T\": " + std::to_string(T) + ", \"backend\": \"" + (ngl > 0 ? "gpu" : "cpu") + "\", \"kv\": \"" + ctk + "/" + ctv + "\",\n";
    auto row = [&](const char * kind, int x, long r0, long r1, const std::vector<float> & v) {
        const metrics self = compare(v.data(), full.data(), r0, r1, V);
        printf("%-7s %5d  rows [%4ld,%4ld)  vs full prefill: max %.2e nrmse %.2e top1 %.4f kl %.1e nonfinite %ld",
               kind, x, r0, r1, self.max_abs, self.nrmse, self.top1, self.kl_mean, self.nonfinite);
        std::string s = "  {\"kind\": \"" + std::string(kind) + "\", \"n\": " + std::to_string(x) + ", \"r0\": " + std::to_string(r0) +
                        ", \"r1\": " + std::to_string(r1) + ", \"vs_full\": " + mjson(self);
        if (!ref.empty()) {
            const metrics m = compare(v.data(), ref.data(), r0, r1, V);
            printf(" | vs ref: max %.2e nrmse %.2e top1 %.4f kl %.1e", m.max_abs, m.nrmse, m.top1, m.kl_mean);
            s += ", \"vs_ref\": " + mjson(m);
        }
        if (!ref32.empty()) {
            const metrics m = compare(v.data(), ref32.data(), r0, r1, V);
            printf(" | vs fp32: kl %.1e top1 %.4f", m.kl_mean, m.top1);
            s += ", \"vs_ref_fp32\": " + mjson(m);
        }
        printf("\n");
        fflush(stdout);
        return s + "}";
    };

    std::vector<std::string> items;
    items.push_back(row("full", T, 0, T, full));
    for (int L : parse_list(lengths_s)) {
        if (L < 1 || L > T) continue;
        run(L, L, cur);
        items.push_back(row("length", L, 0, L, cur));
    }
    for (int S : parse_list(splits_s)) {
        if (S < 1 || S >= T) continue;
        run(S, T, cur);
        // rows [S, T) come from decode steps; rows [0, S) from the split's prefill
        items.push_back(row("split", S, S, T, cur));
    }
    js += " \"runs\": [\n";
    for (size_t i = 0; i < items.size(); ++i) js += items[i] + (i + 1 < items.size() ? ",\n" : "\n");
    js += " ]\n}\n";
    if (!json_path.empty()) {
        FILE * f = fopen(json_path.c_str(), "w");
        fputs(js.c_str(), f);
        fclose(f);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
