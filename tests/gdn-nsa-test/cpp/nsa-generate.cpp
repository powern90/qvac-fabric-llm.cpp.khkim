// Greedy generation over a list of prompts (benchmark runner for the gdn-nsa port).
//
// usage: nsa-generate -m model.gguf -i prompts.txt -o out.txt [-ngl N] [-t threads] [-c n_ctx] [-n max_lines]
//                     [-ctk f16] [-ctv f16]
//   prompts.txt: one sample per line, "id max_new tok1 tok2 ..." (tokens include BOS)
//   out.txt:     one line per sample, "id gen1 gen2 ..." (greedy argmax; stops after EOS id 2, which is written)
// Per sample: llama_memory_clear, prefill all prompt tokens as one ubatch (n_batch = n_ubatch = 4096),
// then decode one token at a time.
#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_path, ctk = "f16", ctv = "f16";
    int ngl = 0, n_threads = 32, n_ctx = 4352, max_lines = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "-m")   model_path = next();
        else if (a == "-i")   in_path    = next();
        else if (a == "-o")   out_path   = next();
        else if (a == "-ngl") ngl        = std::stoi(next());
        else if (a == "-t")   n_threads  = std::stoi(next());
        else if (a == "-c")   n_ctx      = std::stoi(next());
        else if (a == "-n")   max_lines  = std::stoi(next());
        else if (a == "-ctk") ctk        = next();
        else if (a == "-ctv") ctv        = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    const int n_ub = 4096;
    const llama_token EOS = 2;

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
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ub;
    cparams.n_ubatch        = n_ub;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k          = kv_type(ctk);
    cparams.type_v          = kv_type(ctv);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(n_ub, 0, 1);

    std::ifstream fin(in_path);
    FILE * fout = fopen(out_path.c_str(), "w");
    if (!fin || !fout) { fprintf(stderr, "cannot open %s / %s\n", in_path.c_str(), out_path.c_str()); return 1; }
    std::string line;
    int n_done = 0;
    const auto t_start = std::chrono::steady_clock::now();
    while (std::getline(fin, line) && (max_lines < 0 || n_done < max_lines)) {
        std::istringstream ss(line);
        std::string id;
        int max_new = 0;
        if (!(ss >> id >> max_new)) continue;
        std::vector<llama_token> toks;
        long v;
        while (ss >> v) toks.push_back((llama_token) v);
        const int T = (int) toks.size();
        if (T > n_ub || T + max_new > n_ctx) { fprintf(stderr, "%s: T = %d does not fit\n", id.c_str(), T); return 1; }

        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = T;
        for (int i = 0; i < T; ++i) {
            batch.token[i] = toks[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == T - 1;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "%s: prefill failed\n", id.c_str()); return 1; }
        std::vector<llama_token> gen;
        int pos = T;
        for (int s = 0; s < max_new; ++s) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int j = 1; j < V; ++j) if (lg[j] > lg[best]) best = j;
            gen.push_back(best);
            if (best == EOS || s + 1 == max_new) break;
            batch.n_tokens = 1;
            batch.token[0] = best;
            batch.pos[0] = pos++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "%s: decode failed\n", id.c_str()); return 1; }
        }
        fprintf(fout, "%s", id.c_str());
        for (llama_token t : gen) fprintf(fout, " %d", t);
        fprintf(fout, "\n");
        fflush(fout);
        ++n_done;
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "[%d] %s T=%d gen=%zu %.2fs (total %.1fs)\n", n_done, id.c_str(), T, gen.size(), dt, tt);
    }
    fclose(fout);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
