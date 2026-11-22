// Runs a gdn-nsa GGUF on a fixed token list and dumps logits (+ NSA block indices) for comparison with the
// PyTorch reference.
//   nsa-dump -m model.gguf -t tokens.txt -o out_prefix [--mode prefill|decode|chunk] [--chunk N]
//            [-ngl N] [-fa on|off] [-ctk TYPE] [-ctv TYPE] [--gen N] [--idx]
// out_prefix.logits.bin : float32 [T, n_vocab] logits of every input position
// out_prefix.gen.txt    : greedy continuation (--gen N)
// out_prefix.idx.bin    : int32 [n_nsa_layers, T, Hkv, S] selected blocks (--idx)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct idx_capture {
    bool hidden = false;
    std::map<int, std::vector<float>> h; // layer -> [T, n_embd]
    // layer -> per-token rows [Hkv*S] appended in decode order
    std::map<int, std::vector<int32_t>> rows;
    int64_t row_len = 0;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (idx_capture *) user_data;
    const char * hpfx = "l_out-";
    if (cap->hidden && strncmp(t->name, hpfx, strlen(hpfx)) == 0 && t->type == GGML_TYPE_F32) {
        if (ask) return true;
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto & dst = cap->h[atoi(t->name + strlen(hpfx))];
        dst.insert(dst.end(), buf.begin(), buf.end());
        return true;
    }
    const char * pfx = "nsa_idx-";
    if (strncmp(t->name, pfx, strlen(pfx)) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    const int il = atoi(t->name + strlen(pfx));
    std::vector<int32_t> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    cap->row_len = t->ne[0] * t->ne[1];
    auto & dst = cap->rows[il];
    dst.insert(dst.end(), buf.begin(), buf.end());
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, tok_path, out = "out", mode = "prefill", fa = "auto", ctk = "f16", ctv = "f16";
    int ngl = 0, n_gen = 0, chunk = 512;
    bool dump_idx = false, hidden = false, retrim = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model_path = next();
        else if (a == "-t") tok_path = next();
        else if (a == "-o") out = next();
        else if (a == "--mode") mode = next();
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "-ngl") ngl = std::stoi(next());
        else if (a == "-fa") fa = next();
        else if (a == "-ctk") ctk = next();
        else if (a == "-ctv") ctv = next();
        else if (a == "--gen") n_gen = std::stoi(next());
        else if (a == "--idx") dump_idx = true;
        else if (a == "--retrim") retrim = true;
        else if (a == "--hidden") { dump_idx = true; hidden = true; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    std::vector<llama_token> toks;
    {
        std::ifstream f(tok_path);
        long v;
        while (f >> v) toks.push_back((llama_token) v);
    }
    const int T = (int) toks.size();
    if (mode == "prefill") chunk = T;
    if (mode == "decode")  chunk = 1;

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    idx_capture cap;
    cap.hidden = hidden;
    auto cparams = llama_context_default_params();
    cparams.n_ctx     = 4096;
    cparams.n_batch   = std::max(chunk, 1);
    cparams.n_ubatch  = std::max(chunk, 1);
    cparams.n_seq_max = 1;
    auto kv_type = [](const std::string & s) {
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            const char * name = ggml_type_name((ggml_type) t); // NULL in the gaps of the enum
            if (name && s == name) return (ggml_type) t;
        }
        fprintf(stderr, "unknown cache type %s\n", s.c_str());
        exit(1);
    };
    cparams.type_k = kv_type(ctk);
    cparams.type_v = kv_type(ctv);
    cparams.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (dump_idx) {
        cparams.cb_eval = cb_eval;
        cparams.cb_eval_user_data = &cap;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> logits((size_t) T * n_vocab);

    llama_batch batch = llama_batch_init(std::max(chunk, 1), 0, 1);
    for (int p0 = 0; p0 < T; p0 += chunk) {
        const int n = std::min(chunk, T - p0);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = toks[p0 + i];
            batch.pos[i] = p0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", p0); return 1; }
        for (int i = 0; i < n; ++i) {
            memcpy(&logits[(size_t)(p0 + i) * n_vocab], llama_get_logits_ith(ctx, i), sizeof(float) * n_vocab);
        }
    }
    {
        FILE * f = fopen((out + ".logits.bin").c_str(), "wb");
        fwrite(logits.data(), sizeof(float), logits.size(), f);
        fclose(f);
    }
    if (dump_idx) {
        FILE * f = fopen((out + ".idx.bin").c_str(), "wb");
        for (auto & [il, v] : cap.rows) {
            fwrite(v.data(), sizeof(int32_t), std::min<size_t>(v.size(), (size_t) T * cap.row_len), f);
        }
        fclose(f);
        fprintf(stderr, "idx: %zu layers, row_len %lld\n", cap.rows.size(), (long long) cap.row_len);
        cap.rows.clear();
        if (hidden) {
            FILE * fh = fopen((out + ".hidden.bin").c_str(), "wb");
            for (auto & [il, v] : cap.h) fwrite(v.data(), sizeof(float), v.size(), fh);
            fclose(fh);
            fprintf(stderr, "hidden: %zu layers\n", cap.h.size());
            cap.h.clear();
            cap.hidden = false;
        }
    }
    if (retrim) {
        // drop the second half of the sequence from the cache and decode it again: the logits must not change
        const int p0 = T / 2;
        bool p0_done = false;
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1)) {
            printf("retrim: seq_rm of a partial range is rejected by the memory (recurrent state cannot roll back)\n");
            p0_done = true;
        }
        double maxd = 0.0;
        for (int q0 = p0; !p0_done && q0 < T; q0 += chunk) {
            const int n = std::min(chunk, T - q0);
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = toks[q0 + i];
                batch.pos[i] = q0 + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "retrim decode failed at %d\n", q0); return 1; }
            for (int i = 0; i < n; ++i) {
                const float * a = &logits[(size_t)(q0 + i) * n_vocab];
                const float * b = llama_get_logits_ith(ctx, i);
                for (int j = 0; j < n_vocab; ++j) maxd = std::max(maxd, (double) std::fabs(a[j] - b[j]));
            }
        }
        if (!p0_done) printf("retrim: re-decoded positions [%d, %d) after seq_rm, max |logit diff| vs first pass = %.3e\n", p0, T, maxd);
    }
    if (n_gen > 0) {
        std::string gen;
        llama_token cur = 0;
        {
            const float * l = &logits[(size_t)(T - 1) * n_vocab];
            int best = 0;
            for (int j = 1; j < n_vocab; ++j) if (l[j] > l[best]) best = j;
            cur = best;
        }
        for (int s = 0; s < n_gen; ++s) {
            gen += std::to_string(cur) + " ";
            if (s == n_gen - 1) break;
            batch.n_tokens = 1;
            batch.token[0] = cur;
            batch.pos[0] = T + s;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "gen decode failed\n"); return 1; }
            const float * l = llama_get_logits_ith(ctx, 0);
            int best = 0;
            for (int j = 1; j < n_vocab; ++j) if (l[j] > l[best]) best = j;
            cur = best;
        }
        std::ofstream(out + ".gen.txt") << gen << "\n";
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
