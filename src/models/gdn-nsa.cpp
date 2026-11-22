// Hybrid Gated DeltaNet + Native Sparse Attention (fla HybridGatedDeltaNetForCausalLM, NSA attention layers)
//
// GDN layers: same math as the Qwen3.5 linear-attention layers (fused qkv + conv, l2-normed q/k, gated rmsnorm).
// NSA layers: o = sig(g_cmp)*o_cmp + sig(g_slc)*o_slc + sig(g_swa)*o_swa (arXiv:2502.11089), where
//   o_cmp: attention over the mean K/V of the blocks completed by the query position (0 if none)
//   o_slc: attention over the top-n blocks picked by ggml_nsa_select (ggml_nsa_sel_attn)
//   o_swa: sliding window attention over the last nsa_window tokens
// All three branches read the regular (full) KV cache; the block means are recomputed from the cache view.
// The cache may be F16/F32 or quantized (Q8_0/Q4_0, optionally with the Hadamard K/V rotation of llama_kv_cache).
// Limitation: the block math needs cell index == position, i.e. a single sequence stored from cell 0.

#include "models.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

// host-built NSA inputs shared by all NSA layers of a graph
class llm_graph_input_nsa : public llm_graph_input_i {
public:
    llm_graph_input_nsa(uint32_t block_size, uint32_t window) : block_size(block_size), window(window) {}

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(cmp_mask->ne[1] == n_tokens);

        {
            const int64_t nb1 = cmp_mask->ne[0];
            const int64_t nb  = nb1 - 1; // last column: zero K/V for queries without a complete block

            std::vector<float> data(nb1*n_tokens, -INFINITY);
            for (int64_t i = 0; i < n_tokens; ++i) {
                const int64_t p = ubatch->pos[i];
                for (int64_t b = 0; b < nb && (b + 1)*block_size - 1 <= p; ++b) {
                    data[i*nb1 + b] = 0.0f;
                }
                if (p < (int64_t) block_size - 1) {
                    data[i*nb1 + nb] = 0.0f;
                }
            }
            ggml_backend_tensor_set(cmp_mask, data.data(), 0, ggml_nbytes(cmp_mask));
        }

        if (swa_idx) {
            // the blocks that overlap the window [p - window + 1, p], newest first, -1 if before block 0
            const int64_t nw = swa_idx->ne[0];
            const int64_t nh = swa_idx->ne[1];

            std::vector<int32_t> data(nw*nh*n_tokens);
            for (int64_t i = 0; i < n_tokens; ++i) {
                const int64_t cur = ubatch->pos[i] / block_size;
                for (int64_t h = 0; h < nh; ++h) {
                    for (int64_t w = 0; w < nw; ++w) {
                        data[(i*nh + h)*nw + w] = cur - w >= 0 ? (int32_t) (cur - w) : -1;
                    }
                }
            }
            ggml_backend_tensor_set(swa_idx, data.data(), 0, ggml_nbytes(swa_idx));
        }

        if (swa_mask) {
            const int64_t n_kv = swa_mask->ne[0];

            std::vector<float> data(n_kv*n_tokens, -INFINITY);
            for (int64_t i = 0; i < n_tokens; ++i) {
                const int64_t p  = ubatch->pos[i];
                const int64_t j0 = std::max<int64_t>(0, p - (int64_t) window + 1);
                const int64_t j1 = std::min<int64_t>(p, n_kv - 1);
                for (int64_t j = j0; j <= j1; ++j) {
                    data[i*n_kv + j] = 0.0f;
                }
            }
            ggml_backend_tensor_set(swa_mask, data.data(), 0, ggml_nbytes(swa_mask));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        // the masks are refilled in set_input, only their shapes have to match
        const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);
        const int64_t n_kv = mctx->get_attn()->get_n_kv();
        bool res = cmp_mask->ne[0] == n_kv / block_size + 1 && cmp_mask->ne[1] == params.ubatch.n_tokens;
        if (swa_mask) {
            res &= swa_mask->ne[0] == n_kv;
        }
        return res;
    }

    ggml_tensor * cmp_mask = nullptr; // F32 [NB + 1, n_tokens]
    ggml_tensor * swa_idx  = nullptr; // I32 [n_win_blocks, Hkv, n_tokens], sparse sliding window
    ggml_tensor * swa_mask = nullptr; // F32 [n_kv, n_tokens], dense reference path (LLAMA_NSA_DENSE_SWA=1)

    const uint32_t block_size;
    const uint32_t window;
};

} // namespace

void llama_model_gdn_nsa::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, true);

    ml.get_key(LLM_KV_ATTENTION_NSA_BLOCK_SIZE,  hparams.nsa_block_size);
    ml.get_key(LLM_KV_ATTENTION_NSA_BLOCK_COUNT, hparams.nsa_block_count);
    ml.get_key(LLM_KV_ATTENTION_NSA_WINDOW,      hparams.nsa_window);

    // the GDN layers reuse the Qwen3.5 code path with equal k/v head sizes and as many k as v heads
    if (hparams.ssm_dt_rank == 0 || hparams.ssm_d_inner != hparams.ssm_d_state * hparams.ssm_dt_rank ||
        hparams.ssm_n_group != hparams.ssm_dt_rank) {
        throw std::runtime_error("gdn-nsa: only head_v_dim == head_k_dim and num_v_heads == num_heads are supported");
    }
    if (hparams.nsa_block_size == 0 || 256 % hparams.nsa_block_size != 0 || hparams.nsa_block_count == 0) {
        throw std::runtime_error("gdn-nsa: nsa.block_size must divide 256 and nsa.block_count must be > 0");
    }

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_350M; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gdn_nsa::load_arch_tensors(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t value_dim  = hparams.ssm_d_inner;
    const int64_t head_v_dim = value_dim / n_v_heads;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);

        if (hparams.is_recr(il)) {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, conv_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { n_v_heads }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { n_v_heads }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, 0);
        } else {
            layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", il), { n_embd, n_embd_head_k * n_head }, 0);
            layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", il), { n_embd, n_embd_k_gqa }, 0);
            layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", il), { n_embd, n_embd_v_gqa }, 0);
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", il), { n_embd_head_v * n_head, n_embd }, 0);
            // per head gates in (cmp, slc, swa) order
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, 3 * n_head }, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gdn_nsa::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gdn_nsa::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    // the NSA block math maps KV cells to positions 1:1
    // llama_init_from_model rejects n_seq_max > 1 for this arch with an error; this is the invariant
    GGML_ASSERT(cparams.n_seq_max == 1 && "gdn-nsa: only a single sequence is supported");

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    // no fused kv rows: the NSA masks index KV cells directly
    auto * inp = build_inp_mem_hybrid(false);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_gdn(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_nsa(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_inp = cur;

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// same as the Qwen3.5 linear attention layer (non-deferred path)
ggml_tensor * llama_model_gdn_nsa::graph::build_layer_gdn(llm_graph_input_rs * inp, ggml_tensor * cur, int il) {
    const auto & layer    = model.layers[il];
    const auto * mctx_cur = inp->mctx;

    // the qvac deferred commit is only wired for qwen35 (see llama_memory_recurrent)
    GGML_ASSERT(inp->rec_idx == nullptr);

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = build_lora_mm(layer.wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(layer.wqkv_gate, cur);
    cb(z, "z", il);

    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(layer.ssm_alpha, cur);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);

    // g = -exp(A_log) * softplus(a + dt_bias), ssm_a holds -exp(A_log)
    ggml_tensor * gate = ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt));
    gate = ggml_mul(ctx0, gate, layer.ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(gate, "gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    const int64_t conv_kernel_size = layer.ssm_conv1d->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv_out, "conv_output_silu", il);

    ggml_tensor * state = build_rs_ssm(inp, ssm_states_all, n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const size_t  nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_out->type, head_k_dim * num_k_heads));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    q_conv = build_gdn_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // gated rmsnorm: rmsnorm(o) * w * silu(z)
    ggml_tensor * z_4d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * normed = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    normed = ggml_mul(ctx0, normed, ggml_silu(ctx0, z_4d));

    normed = ggml_reshape_2d(ctx0, normed, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);

    cur = build_lora_mm(layer.ssm_out, normed);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_gdn_nsa::graph::build_layer_nsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t D   = hparams.n_embd_head_k(il);
    const int64_t Dv  = hparams.n_embd_head_v(il);
    const int64_t Hq  = hparams.n_head(il);
    const int64_t Hkv = hparams.n_head_kv(il);
    const int64_t bs  = hparams.nsa_block_size;

    const float scale = 1.0f / sqrtf(float(D));

    ggml_tensor * Qcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, cur), D,  Hq,  n_tokens);
    ggml_tensor * Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, cur), D,  Hkv, n_tokens);
    ggml_tensor * Vcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, cur), Dv, Hkv, n_tokens);

    // [3, Hq, n_tokens]: (cmp, slc, swa)
    ggml_tensor * gates = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, cur));
    gates = ggml_reshape_3d(ctx0, gates, 3, Hq, n_tokens);
    cb(gates, "nsa_gates", il);

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    GGML_ASSERT(!cparams.training);
    GGML_ASSERT(!inp->self_attn_mask.rows);

    // Hadamard rotation of the head dim ahead of a quantized cache (same as build_attn). q.k and the block means are
    // invariant under the orthogonal rotation, and the three branches are linear in V with per-head scalar gates, so
    // the gated sum is rotated back once below.
    if (inp->self_k_rot) {
        Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp->self_k_rot);
        Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot);
    }

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Vcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp->get_v_idxs(), il));

    ggml_tensor * k = mctx_cur->get_k(ctx0, il); // [D, Hkv, n_kv, 1]
    ggml_tensor * v = mctx_cur->get_v(ctx0, il); // [Dv, Hkv, n_kv, 1], or [n_kv, Hkv, Dv, 1] if transposed

    const bool v_trans = v->nb[1] > v->nb[2];

    const int64_t n_kv = k->ne[2];
    GGML_ASSERT(k->ne[3] == 1 && n_kv % bs == 0);
    const int64_t n_blk = n_kv / bs;

    // V with one row per token: [Dv, Hkv, n_kv]
    ggml_tensor * v_rows = v_trans ? ggml_permute(ctx0, v, 2, 1, 0, 3) : v;

    // reference path: dense sliding window over the whole KV cache with a mask
    static const bool dense_swa = [] {
        const char * env = getenv("LLAMA_NSA_DENSE_SWA");
        return env && atoi(env) != 0;
    }();

    if (!nsa_cmp_mask) {
        auto inp_nsa = std::make_unique<llm_graph_input_nsa>(hparams.nsa_block_size, hparams.nsa_window);
        inp_nsa->cmp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk + 1, n_tokens);
        ggml_set_input(inp_nsa->cmp_mask);
        if (hparams.nsa_window == 0) {
            // no sliding window branch
        } else if (dense_swa) {
            inp_nsa->swa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_tokens);
            ggml_set_input(inp_nsa->swa_mask);
        } else {
            const int64_t n_win_blk = (hparams.nsa_window + bs - 2) / bs + 1;
            inp_nsa->swa_idx = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, n_win_blk, Hkv, n_tokens);
            ggml_set_input(inp_nsa->swa_idx);
        }
        nsa_cmp_mask = inp_nsa->cmp_mask;
        nsa_swa_mask = inp_nsa->swa_mask;
        nsa_swa_idx  = inp_nsa->swa_idx;
        res->add_input(std::move(inp_nsa));
    }
    GGML_ASSERT(nsa_cmp_mask->ne[0] == n_blk + 1);

    // q [D, Hq, T], k [D, Hkv, N], v_t [N, Dv, Hkv], mask [N, T] -> [Dv, Hq, T]
    auto dense_attn = [&](ggml_tensor * q, ggml_tensor * kk, ggml_tensor * v_t, ggml_tensor * mask) {
        ggml_tensor * kq = ggml_mul_mat(ctx0, ggml_permute(ctx0, kk, 0, 2, 1, 3), ggml_permute(ctx0, q, 0, 2, 1, 3));
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx0, kq, mask, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v_t, kq);
        return ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
    };

    // mean over each block of bs tokens: [Dx, H, n_kv] -> [Dx, H, n_blk]
    auto block_mean = [&](ggml_tensor * x) {
        const int64_t dx = x->ne[0];
        const int64_t h  = x->ne[1];
        ggml_tensor * xf = ggml_cast(ctx0, x, GGML_TYPE_F32);
        xf = ggml_reshape_3d(ctx0, xf, dx*h, bs, n_blk);
        xf = ggml_cont(ctx0, ggml_permute(ctx0, xf, 1, 0, 2, 3));
        xf = ggml_mean(ctx0, xf);
        return ggml_reshape_3d(ctx0, xf, dx, h, n_blk);
    };

    ggml_tensor * k_cmp = block_mean(k);
    ggml_tensor * v_cmp = block_mean(v_rows);
    cb(k_cmp, "nsa_k_cmp", il);

    // compressed attention; an extra zero block catches the queries with no complete block (output 0)
    ggml_tensor * k_cmp_z = ggml_pad(ctx0, k_cmp, 0, 0, 1, 0);
    ggml_tensor * v_cmp_z = ggml_pad(ctx0, v_cmp, 0, 0, 1, 0);
    ggml_tensor * o_cmp = dense_attn(Qcur, k_cmp_z, ggml_cont(ctx0, ggml_permute(ctx0, v_cmp_z, 1, 2, 0, 3)), nsa_cmp_mask);
    cb(o_cmp, "nsa_o_cmp", il);

    // block selection + selected attention
    ggml_tensor * idx = ggml_nsa_select(ctx0, Qcur, k_cmp, inp_pos, hparams.nsa_block_size, hparams.nsa_block_count, scale);
    cb(idx, "nsa_idx", il);

    ggml_tensor * o_slc = ggml_nsa_sel_attn(ctx0, Qcur, k, v_rows, idx, inp_pos, hparams.nsa_block_size, scale);
    cb(o_slc, "nsa_o_slc", il);

    // sliding window: the same sparse kernel over the blocks that overlap the window (window 0: no such branch)
    ggml_tensor * o_swa = nullptr;
    if (hparams.nsa_window == 0) {
        // nothing
    } else if (dense_swa) {
        // a quantized V (rows of blocks) cannot be permuted out of dim 0: dequantize it first
        ggml_tensor * v_mm = v_trans ? ggml_permute(ctx0, v, 0, 2, 1, 3)
            : ggml_cont(ctx0, ggml_permute(ctx0, ggml_is_quantized(v->type) ? ggml_cast(ctx0, v, GGML_TYPE_F32) : v, 1, 2, 0, 3));
        o_swa = dense_attn(Qcur, k, v_mm, nsa_swa_mask);
    } else {
        o_swa = ggml_nsa_sel_attn_ext(ctx0, Qcur, k, v_rows, nsa_swa_idx, inp_pos, hparams.nsa_block_size, scale, hparams.nsa_window);
    }

    auto gate = [&](int i) {
        return ggml_cont(ctx0, ggml_view_3d(ctx0, gates, 1, Hq, n_tokens, gates->nb[1], gates->nb[2], i*ggml_element_size(gates)));
    };

    ggml_tensor * o = ggml_mul(ctx0, o_cmp, gate(0));
    o = ggml_add(ctx0, o, ggml_mul(ctx0, o_slc, gate(1)));
    if (o_swa) {
        cb(o_swa, "nsa_o_swa", il);
        o = ggml_add(ctx0, o, ggml_mul(ctx0, o_swa, gate(2)));
    }
    if (inp->self_v_rot) {
        o = llama_mul_mat_hadamard(ctx0, o, inp->self_v_rot);
    }
    cb(o, "nsa_out", il);

    o = ggml_reshape_2d(ctx0, o, Dv*Hq, n_tokens);

    cur = build_lora_mm(layer.wo, o);
    cb(cur, "attn_output", il);

    return cur;
}
