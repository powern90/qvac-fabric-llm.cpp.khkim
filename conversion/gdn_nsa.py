from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("HybridGatedDeltaNetForCausalLM")
class GdnNsaModel(TextModel):
    """Hybrid Gated DeltaNet + Native Sparse Attention (fla, idiap/hybrid-linear-sparse-attention).

    GDN layers reuse the Qwen3.5 linear-attention tensor layout (fused qkv, fused conv, ssm_a = -exp(A_log)).
    Norms are plain RMSNorm (x * w): unlike Qwen3-Next/3.5 no +1 is applied.
    """
    model_arch = gguf.MODEL_ARCH.GDN_NSA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attn = self.hparams["attn"]
        assert attn.get("attn_type") == "nsa", "only NSA attention layers are supported"
        self.attn_layers = set(attn["layers"])
        self._pending: dict[tuple[int, str], dict[str, Tensor]] = {}

    def set_vocab(self):
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
            return

        # tokenizer.json only (SPM-style BPE with byte fallback). LlamaHfVocab gives every token the same
        # score, which makes the llama.cpp SPM tokenizer merge in the wrong order. Use the BPE merge rank
        # as the score instead, so that the highest-priority merge wins as in the HF tokenizer.
        import json
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens, scores, toktypes = [], [], []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)
        assert len(tokens) == vocab.vocab_size

        with open(self.dir_model / "tokenizer.json", encoding="utf-8") as f:
            model = json.load(f)["model"]
        assert model["type"] == "BPE"
        tok_to_id = model["vocab"]
        merged_score: dict[int, float] = {}
        for rank, m in enumerate(model["merges"]):
            a, b = m if isinstance(m, list) else m.split(" ", 1)
            tid = tok_to_id.get(a + b)
            if tid is not None and tid not in merged_score:
                merged_score[tid] = -float(rank)
        for tid in range(len(tokens)):
            if toktypes[tid] == gguf.TokenType.NORMAL:
                scores[tid] = merged_score.get(tid, 0.0)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        attn = hp["attn"]

        # TextModel writes key/value_length from the top-level head_dim, which is the GDN head size (128). The
        # attention key/value length is the NSA head size (attn.head_dim = 64), written below, so hide the GDN one
        # from the base class instead of overwriting the keys (GGUFWriter warns on duplicated keys).
        gdn_head_dim = hp.pop("head_dim")
        try:
            super().set_gguf_parameters()  # also writes the rms eps from norm_eps
        finally:
            hp["head_dim"] = gdn_head_dim

        n_ff = hp.get("intermediate_size")
        if n_ff is None:
            # same rule as fla GatedMLP
            n_ff = int(hp["hidden_size"] * hp["hidden_ratio"] * 2 / 3)
            n_ff = 256 * ((n_ff + 256 - 1) // 256)
        self.gguf_writer.add_feed_forward_length(n_ff)

        # attention (NSA) layers
        self.gguf_writer.add_head_count(attn["num_heads"])
        self.gguf_writer.add_head_count_kv(attn["num_kv_heads"])
        self.gguf_writer.add_key_length(attn["head_dim"])
        self.gguf_writer.add_value_length(attn["head_dim"])
        self.gguf_writer.add_rope_dimension_count(attn["head_dim"])
        self.gguf_writer.add_rope_freq_base(attn["rope_theta"])
        self.gguf_writer.add_nsa_block_size(attn["block_size"])
        self.gguf_writer.add_nsa_block_count(attn["block_counts"])
        self.gguf_writer.add_nsa_window(attn["window_size"])

        # gated delta net layers
        n_k_heads = hp["num_heads"]
        n_v_heads = hp.get("num_v_heads") or n_k_heads
        head_k_dim = hp["head_dim"]
        head_v_dim = int(head_k_dim * hp["expand_v"])
        assert hp.get("use_gate", True) and hp.get("use_short_conv", True)
        # the GDN graph is the Qwen3.5 one with equal k/v head sizes and no grouped value heads
        assert head_v_dim == head_k_dim, "gdn-nsa: expand_v != 1 is not supported"
        assert n_v_heads == n_k_heads, "gdn-nsa: num_v_heads != num_heads is not supported"
        if head_k_dim > 128:
            logger.warning("GDN head_dim %d > 128: the CUDA gated delta net kernel only has head sizes <= 128", head_k_dim)
        assert not hp.get("allow_neg_eigval", False)
        self.gguf_writer.add_ssm_conv_kernel(hp["conv_size"])
        self.gguf_writer.add_ssm_state_size(head_k_dim)
        self.gguf_writer.add_ssm_group_count(n_k_heads)
        self.gguf_writer.add_ssm_time_step_rank(n_v_heads)
        self.gguf_writer.add_ssm_inner_size(head_v_dim * n_v_heads)

        self.gguf_writer.add_recurrent_layers([i not in self.attn_layers for i in range(self.block_count)])

    def _fuse(self, bid: int, kind: str, part: str, data: Tensor, parts: tuple[str, ...]) -> Tensor | None:
        buf = self._pending.setdefault((bid, kind), {})
        buf[part] = data
        if len(buf) < len(parts):
            return None
        del self._pending[(bid, kind)]
        return torch.cat([buf[p] for p in parts], dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        fmt = self.format_tensor_name

        if name == "model.embeddings.weight":
            yield fmt(T.TOKEN_EMBD), data_torch
            return
        if name == "model.norm.weight":
            yield fmt(T.OUTPUT_NORM), data_torch
            return
        if name == "lm_head.weight":
            if not self.hparams.get("tie_word_embeddings", False):
                yield fmt(T.OUTPUT), data_torch
            return

        assert bid is not None, name
        sub = name.split(f"model.layers.{bid}.", 1)[1]

        common = {
            "attn_norm.weight":     (T.ATTN_NORM, ".weight"),
            "mlp_norm.weight":      (T.FFN_NORM,  ".weight"),
            "mlp.gate_proj.weight": (T.FFN_GATE,  ".weight"),
            "mlp.up_proj.weight":   (T.FFN_UP,    ".weight"),
            "mlp.down_proj.weight": (T.FFN_DOWN,  ".weight"),
        }
        if sub in common:
            t, suffix = common[sub]
            yield fmt(t, bid, suffix), data_torch
            return

        if bid in self.attn_layers:
            nsa = {
                "attn.q_proj.weight": T.ATTN_Q,
                "attn.k_proj.weight": T.ATTN_K,
                "attn.v_proj.weight": T.ATTN_V,
                "attn.o_proj.weight": T.ATTN_OUT,
                "attn.g_proj.weight": T.ATTN_GATE,  # [n_head * 3] gates in (cmp, slc, swa) order per head
            }
            if sub not in nsa:
                raise ValueError(f"unexpected NSA tensor {name}")
            yield fmt(nsa[sub], bid, ".weight"), data_torch
            return

        # gated delta net layer
        if sub in ("attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"):
            fused = self._fuse(bid, "proj", sub.split(".")[1][0], data_torch, ("q", "k", "v"))
            if fused is not None:
                yield fmt(T.ATTN_QKV, bid, ".weight"), fused
            return
        if sub in ("attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight"):
            # [C, 1, d_conv] -> [C, d_conv]
            fused = self._fuse(bid, "conv", sub.split(".")[1][0], data_torch.squeeze(1), ("q", "k", "v"))
            if fused is not None:
                yield fmt(T.SSM_CONV1D, bid, ".weight"), fused
            return

        gdn = {
            "attn.g_proj.weight": (T.ATTN_GATE,  ".weight"),
            "attn.b_proj.weight": (T.SSM_BETA,   ".weight"),
            "attn.a_proj.weight": (T.SSM_ALPHA,  ".weight"),
            "attn.dt_bias":       (T.SSM_DT,     ".bias"),
            "attn.o_norm.weight": (T.SSM_NORM,   ".weight"),
            "attn.o_proj.weight": (T.SSM_OUT,    ".weight"),
        }
        if sub == "attn.A_log":
            yield fmt(T.SSM_A, bid, ""), -torch.exp(data_torch.float())
            return
        if sub not in gdn:
            raise ValueError(f"unexpected GDN tensor {name}")
        t, suffix = gdn[sub]
        yield fmt(t, bid, suffix), data_torch

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._pending:
            raise ValueError(f"unfused tensors left: {list(self._pending)}")
