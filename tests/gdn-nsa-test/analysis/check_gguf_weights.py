#!/usr/bin/env python3
"""Level-0 validation for gdn-nsa: is the GGUF an exact conversion of the HF checkpoint?

The HF -> GGUF mapping is re-implemented here from conversion/gdn_nsa.py (the converter is not imported, so a
bug in it is not copied into the check). Checks:

  1. tensors     every GGUF tensor is rebuilt from the HF tensors and compared bitwise (uint32 views for F32)
  2. coverage    every HF tensor is used exactly once (lm_head may be skipped when tied), every GGUF tensor
                 has a recipe, nothing is missing
  3. projection  every 2-D GGUF tensor is read from the raw file bytes with ggml strides (ne[0] = input
                 features, contiguous) and y = W x is compared with the float64 HF reference
                 (per-part products concatenated for the fused tensors); ne[0] must equal the HF in_features
  4. metadata    GGUF KV pairs vs config.json (+ a few weight-shape vs config consistency checks)

Exit code 1 on any failure.

usage (from the repo root):
  PYTHONPATH=gguf-py <python> tests/gdn-nsa-test/analysis/check_gguf_weights.py \\
      --ckpt <HF checkpoint dir> \\
      --gguf <f32 gguf> \\
      --out  l0_weights.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf
from gguf import GGUFReader, GGMLQuantizationType

ARCH = "gdn-nsa"
PROJ_REL_TOL = 1e-12      # float64 products of bitwise-identical weights: only summation order differs
NEG_CONTROL_MIN = 1e-2    # the transposed reading of a square weight must be clearly wrong


# ----------------------------------------------------------------------------------------------------------------
# HF -> GGUF recipes (mirror of conversion/gdn_nsa.py)
# ----------------------------------------------------------------------------------------------------------------

def build_recipes(cfg: dict, hf_names: set[str]):
    """Return (recipes, skipped_hf). recipes: gguf name -> {kind, srcs, layer_type}."""
    tied = bool(cfg.get("tie_word_embeddings", False))
    attn_layers = set(cfg["attn"]["layers"])
    recipes: OrderedDict[str, dict] = OrderedDict()
    skipped: list[str] = []

    def add(gname, kind, srcs, layer_type):
        assert gname not in recipes, gname
        recipes[gname] = {"kind": kind, "srcs": srcs, "layer_type": layer_type}

    add("token_embd.weight", "copy", ["model.embeddings.weight"], "global")
    add("output_norm.weight", "copy", ["model.norm.weight"], "global")
    if tied:
        if "lm_head.weight" in hf_names:
            skipped.append("lm_head.weight")
    else:
        add("output.weight", "copy", ["lm_head.weight"], "global")

    for i in range(cfg["num_hidden_layers"]):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        lt = "nsa" if i in attn_layers else "gdn"
        add(b + "attn_norm.weight", "copy", [p + "attn_norm.weight"], lt)
        add(b + "ffn_norm.weight",  "copy", [p + "mlp_norm.weight"], lt)
        add(b + "ffn_gate.weight",  "copy", [p + "mlp.gate_proj.weight"], lt)
        add(b + "ffn_up.weight",    "copy", [p + "mlp.up_proj.weight"], lt)
        add(b + "ffn_down.weight",  "copy", [p + "mlp.down_proj.weight"], lt)
        if lt == "nsa":
            add(b + "attn_q.weight",      "copy", [p + "attn.q_proj.weight"], lt)
            add(b + "attn_k.weight",      "copy", [p + "attn.k_proj.weight"], lt)
            add(b + "attn_v.weight",      "copy", [p + "attn.v_proj.weight"], lt)
            add(b + "attn_output.weight", "copy", [p + "attn.o_proj.weight"], lt)
            add(b + "attn_gate.weight",   "copy", [p + "attn.g_proj.weight"], lt)
        else:
            add(b + "attn_qkv.weight", "cat",
                [p + "attn.q_proj.weight", p + "attn.k_proj.weight", p + "attn.v_proj.weight"], lt)
            add(b + "ssm_conv1d.weight", "conv_cat",
                [p + "attn.q_conv1d.weight", p + "attn.k_conv1d.weight", p + "attn.v_conv1d.weight"], lt)
            add(b + "attn_gate.weight",  "copy", [p + "attn.g_proj.weight"], lt)
            add(b + "ssm_beta.weight",   "copy", [p + "attn.b_proj.weight"], lt)
            add(b + "ssm_alpha.weight",  "copy", [p + "attn.a_proj.weight"], lt)
            add(b + "ssm_dt.bias",       "copy", [p + "attn.dt_bias"], lt)
            add(b + "ssm_norm.weight",   "copy", [p + "attn.o_norm.weight"], lt)
            add(b + "ssm_out.weight",    "copy", [p + "attn.o_proj.weight"], lt)
            add(b + "ssm_a",             "neg_exp", [p + "attn.A_log"], lt)
    return recipes, skipped


def build_expected(rec: dict, hf: dict[str, torch.Tensor]) -> torch.Tensor:
    ts = [hf[s] for s in rec["srcs"]]
    k = rec["kind"]
    if k == "copy":
        t = ts[0]
    elif k == "cat":
        t = torch.cat(ts, dim=0)
    elif k == "conv_cat":
        t = torch.cat([x.squeeze(1) for x in ts], dim=0)
    elif k == "neg_exp":
        t = -torch.exp(ts[0].float())  # exactly as the converter: torch float32
    else:
        raise ValueError(k)
    return t.contiguous()


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------

def load_hf(ckpt: Path) -> dict[str, torch.Tensor]:
    files = sorted(ckpt.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {ckpt}")
    out: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                assert k not in out, f"duplicate HF tensor {k}"
                out[k] = sf.get_tensor(k)
    return out


def ggml_raw(path: str, t) -> np.ndarray:
    """Flat tensor data straight from the file (independent of the reader's reshape)."""
    if t.tensor_type == GGMLQuantizationType.F32:
        dt = np.float32
    elif t.tensor_type == GGMLQuantizationType.F16:
        dt = np.float16
    else:
        raise TypeError(f"unsupported GGUF tensor type {t.tensor_type.name}")
    return np.memmap(path, dtype=dt, mode="r", offset=int(t.data_offset), shape=(int(t.n_elements),))


def group_key(name: str, layer_type: str) -> str:
    g = re.sub(r"^blk\.\d+\.", "blk.*.", name)
    return f"{g} [{layer_type}]" if layer_type != "global" else g


def fmt_e(v) -> str:
    if v is None:
        return "-"
    return "0" if v == 0 else f"{v:.2e}"


# ----------------------------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------------------------

def check_tensors(reader, recipes, hf):
    rows = []
    for t in reader.tensors:
        rec = recipes.get(t.name)
        ne = [int(x) for x in t.shape]
        row = {"name": t.name, "ne": ne, "dtype": t.tensor_type.name, "kind": None, "srcs": None,
               "shape_ok": False, "max_abs_diff": None, "max_ulp": None, "bitwise_equal": False, "pass": False}
        rows.append(row)
        if rec is None:
            row["error"] = "no recipe for this GGUF tensor"
            continue
        row["kind"], row["srcs"], row["layer_type"] = rec["kind"], rec["srcs"], rec["layer_type"]
        exp_t = build_expected(rec, hf)
        if t.tensor_type == GGMLQuantizationType.F32:
            exp = exp_t.to(torch.float32).numpy()
            uint = np.uint32
        elif t.tensor_type == GGMLQuantizationType.F16:
            exp = exp_t.to(torch.float16).numpy()
            uint = np.uint16
        else:
            row["error"] = f"unsupported GGUF dtype {t.tensor_type.name}"
            continue
        got = np.asarray(t.data)
        row["shape"] = list(got.shape)
        row["expected_shape"] = list(exp.shape)
        row["shape_ok"] = (list(got.shape) == list(exp.shape)) and (ne == list(reversed(exp.shape)))
        if not row["shape_ok"]:
            row["error"] = f"shape mismatch: gguf ne={ne} np={list(got.shape)} expected np={list(exp.shape)}"
            continue
        a, e = got.reshape(-1), exp.reshape(-1)
        row["bitwise_equal"] = bool(np.array_equal(a.view(uint), e.view(uint)))
        d = np.abs(a.astype(np.float64) - e.astype(np.float64))
        row["max_abs_diff"] = float(np.nanmax(d)) if d.size else 0.0
        if not row["bitwise_equal"]:
            ia = a.view(np.int32 if uint is np.uint32 else np.int16).astype(np.int64)
            ie = e.view(np.int32 if uint is np.uint32 else np.int16).astype(np.int64)
            row["max_ulp"] = int(np.max(np.abs(ia - ie)))
            row["n_diff"] = int(np.count_nonzero(a.view(uint) != e.view(uint)))
        row["pass"] = row["shape_ok"] and row["bitwise_equal"]
    return rows


def check_coverage(reader, recipes, skipped, hf, expect_n):
    gguf_names = [t.name for t in reader.tensors]
    gset = set(gguf_names)
    used = Counter()
    for gname, rec in recipes.items():
        if gname in gset:
            for s in rec["srcs"]:
                used[s] += 1
    hf_missing_for_recipe = sorted({s for rec in recipes.values() for s in rec["srcs"] if s not in hf})
    hf_unused = sorted(k for k in hf if used[k] == 0 and k not in skipped)
    hf_multi = sorted(k for k in hf if used[k] > 1)
    dup_gguf = sorted(k for k, c in Counter(gguf_names).items() if c > 1)
    gguf_no_recipe = sorted(gset - set(recipes))
    recipe_not_in_gguf = sorted(set(recipes) - gset)
    tc = reader.fields.get("GGUF.tensor_count")
    tensor_count_field = int(tc.contents()) if tc is not None else None

    tied_lm_head = None
    if "lm_head.weight" in skipped:
        tied_lm_head = bool(torch.equal(hf["lm_head.weight"], hf["model.embeddings.weight"]))

    res = {
        "n_hf_tensors": len(hf),
        "n_hf_used_once": sum(1 for k in hf if used[k] == 1),
        "hf_skipped": skipped,
        "hf_unused": hf_unused,
        "hf_used_more_than_once": hf_multi,
        "hf_missing_for_recipe": hf_missing_for_recipe,
        "n_gguf_tensors": len(gguf_names),
        "gguf_tensor_count_field": tensor_count_field,
        "n_recipes": len(recipes),
        "n_gguf_accounted": len(gset & set(recipes)),
        "gguf_without_recipe": gguf_no_recipe,
        "recipe_missing_in_gguf": recipe_not_in_gguf,
        "gguf_duplicate_names": dup_gguf,
        "expected_n_gguf_tensors": expect_n,
        "skipped_lm_head_equals_embeddings": tied_lm_head,
    }
    checks = {
        "every HF tensor used exactly once (or skipped)": not hf_unused and not hf_multi,
        "every recipe source present in HF": not hf_missing_for_recipe,
        "every GGUF tensor accounted for": not gguf_no_recipe and not dup_gguf,
        "no expected GGUF tensor missing": not recipe_not_in_gguf,
        "GGUF.tensor_count field == #tensors": tensor_count_field == len(gguf_names),
    }
    if expect_n:
        checks[f"#GGUF tensors == {expect_n}"] = len(gguf_names) == expect_n
    if tied_lm_head is not None:
        checks["skipped lm_head == embeddings (tied)"] = tied_lm_head
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


def check_projection(path, reader, recipes, hf, seed, n_vec):
    rng = np.random.default_rng(seed)
    rows = []
    for t in reader.tensors:
        if len(t.shape) != 2 or t.name not in recipes:
            continue
        rec = recipes[t.name]
        ne0, ne1 = int(t.shape[0]), int(t.shape[1])
        srcs = [hf[s] for s in rec["srcs"]]
        if rec["kind"] == "conv_cat":
            parts = [s.squeeze(1) for s in srcs]       # [C, 1, K] -> [C, K]: y[c] = sum_k w[c, k] x[k]
        else:
            parts = list(srcs)                          # Linear / Embedding: [out, in]
        in_f = {int(p.shape[-1]) for p in parts}
        out_f = sum(int(p.shape[0]) for p in parts)
        row = {"name": t.name, "ne": [ne0, ne1], "layer_type": rec["layer_type"],
               "hf_in_features": sorted(in_f), "hf_out_features": out_f,
               "ne0_eq_in": len(in_f) == 1 and ne0 == next(iter(in_f)), "ne1_eq_out": ne1 == out_f,
               "max_rel_err": None, "neg_control_rel_err": None, "pass": False}
        rows.append(row)
        if not (row["ne0_eq_in"] and row["ne1_eq_out"]):
            row["error"] = f"ggml ne={[ne0, ne1]} vs HF in={sorted(in_f)} out={out_f}"
            continue

        # ggml semantics: element (i0, i1) lives at i0*nb0 + i1*nb1, nb0 = type size, nb1 = ne0*nb0 (contiguous);
        # ggml_mul_mat(W, x): y[i1] = sum_i0 W[i0, i1] * x[i0]
        flat = ggml_raw(path, t)
        isz = flat.dtype.itemsize
        W = np.lib.stride_tricks.as_strided(flat, shape=(ne0, ne1), strides=(isz, ne0 * isz), writeable=False)
        X = rng.standard_normal((ne0, n_vec))
        Wd = W.astype(np.float64)
        y_g = Wd.T @ X                                                         # [ne1, n_vec]
        y_ref = np.concatenate([p.double().numpy() @ X for p in parts], axis=0)  # HF: W_hf [out, in] @ x
        scale = float(np.max(np.abs(y_ref))) or 1.0
        row["max_rel_err"] = float(np.max(np.abs(y_g - y_ref)) / scale)
        ok = row["max_rel_err"] <= PROJ_REL_TOL
        if ne0 == ne1:
            # negative control: reading the buffer transposed must give a clearly different y
            y_wrong = Wd @ X
            row["neg_control_rel_err"] = float(np.max(np.abs(y_wrong - y_ref)) / scale)
            ok = ok and row["neg_control_rel_err"] >= NEG_CONTROL_MIN
        if t.name == "token_embd.weight":
            # get_rows semantics: token id j -> row j = flat[j*ne0 : (j+1)*ne0]
            ids = rng.integers(0, ne1, size=64)
            emb = hf[rec["srcs"][0]].numpy()
            gr = all(np.array_equal(np.asarray(flat[j * ne0:(j + 1) * ne0]).view(np.uint32), emb[j].view(np.uint32))
                     for j in ids) if flat.dtype == np.float32 else None
            row["get_rows_bitwise_equal"] = gr
            ok = ok and gr is not False
        row["pass"] = bool(ok)
    return rows


def check_metadata(reader, cfg, hf, recipes):
    fields = reader.fields

    def get(key):
        f = fields.get(key)
        return None if f is None else f.contents()

    def arr_len(key):
        f = fields.get(key)
        return None if f is None else len(f.data)

    rows = []

    def chk(name, expected, actual, cmp=None):
        ok = (cmp(expected, actual) if cmp else expected == actual) if actual is not None else False
        rows.append({"check": name, "expected": expected, "actual": actual, "pass": bool(ok)})

    f32eq = lambda a, b: np.float32(a) == np.float32(b)  # noqa: E731
    k = lambda s: f"{ARCH}.{s}"  # noqa: E731
    attn = cfg["attn"]
    n_layer = cfg["num_hidden_layers"]

    chk("general.architecture", ARCH, get("general.architecture"))
    chk(k("block_count") + " == num_hidden_layers", n_layer, get(k("block_count")))
    chk(k("context_length") + " == max_position_embeddings", cfg["max_position_embeddings"], get(k("context_length")))
    chk(k("embedding_length") + " == hidden_size", cfg["hidden_size"], get(k("embedding_length")))
    chk(k("attention.layer_norm_rms_epsilon") + " == norm_eps", cfg["norm_eps"],
        get(k("attention.layer_norm_rms_epsilon")), f32eq)

    # NSA (attention) layers
    chk(k("attention.head_count") + " == attn.num_heads", attn["num_heads"], get(k("attention.head_count")))
    chk(k("attention.head_count_kv") + " == attn.num_kv_heads", attn["num_kv_heads"], get(k("attention.head_count_kv")))
    chk(k("attention.key_length") + " == attn.head_dim", attn["head_dim"], get(k("attention.key_length")))
    chk(k("attention.value_length") + " == attn.head_dim", attn["head_dim"], get(k("attention.value_length")))
    chk(k("rope.dimension_count") + " == attn.head_dim", attn["head_dim"], get(k("rope.dimension_count")))
    chk(k("rope.freq_base") + " == attn.rope_theta", attn["rope_theta"], get(k("rope.freq_base")), f32eq)
    chk(k("attention.nsa.block_size") + " == attn.block_size", attn["block_size"], get(k("attention.nsa.block_size")))
    chk(k("attention.nsa.block_count") + " == attn.block_counts", attn["block_counts"], get(k("attention.nsa.block_count")))
    chk(k("attention.nsa.window_size") + " == attn.window_size", attn["window_size"], get(k("attention.nsa.window_size")))

    # FFN: fla GatedMLP rule
    n_ff = cfg.get("intermediate_size")
    if n_ff is None:
        n_ff = int(cfg["hidden_size"] * cfg["hidden_ratio"] * 2 / 3)
        n_ff = 256 * ((n_ff + 256 - 1) // 256)
    chk(k("feed_forward_length") + " == fla GatedMLP(hidden, hidden_ratio)", n_ff, get(k("feed_forward_length")))
    hf_ff = sorted({int(hf[f"model.layers.{i}.mlp.gate_proj.weight"].shape[0]) for i in range(n_layer)})
    chk("HF mlp.gate_proj out (all layers) == feed_forward_length", [n_ff], hf_ff)

    # GDN (recurrent) layers
    n_k = cfg["num_heads"]
    n_v = cfg.get("num_v_heads") or n_k
    hk = cfg["head_dim"]
    hv = int(hk * cfg["expand_v"])
    chk(k("ssm.conv_kernel") + " == conv_size", cfg["conv_size"], get(k("ssm.conv_kernel")))
    chk(k("ssm.state_size") + " == head_dim (GDN head_k_dim)", hk, get(k("ssm.state_size")))
    chk(k("ssm.group_count") + " == num_heads", n_k, get(k("ssm.group_count")))
    chk(k("ssm.time_step_rank") + " == num_v_heads", n_v, get(k("ssm.time_step_rank")))
    chk(k("ssm.inner_size") + " == head_v_dim * num_v_heads", hv * n_v, get(k("ssm.inner_size")))
    rec_exp = [i not in set(attn["layers"]) for i in range(n_layer)]
    rl = get(k("attention.recurrent_layers"))
    chk(k("attention.recurrent_layers") + " == not in attn.layers", rec_exp,
        [bool(x) for x in rl] if rl is not None else None)

    # vocab / embeddings
    V = cfg["vocab_size"]
    chk("len(tokenizer.ggml.tokens) == vocab_size", V, arr_len("tokenizer.ggml.tokens"))
    chk("len(tokenizer.ggml.scores) == vocab_size", V, arr_len("tokenizer.ggml.scores"))
    chk("len(tokenizer.ggml.token_type) == vocab_size", V, arr_len("tokenizer.ggml.token_type"))
    tens = {t.name: [int(x) for x in t.shape] for t in reader.tensors}
    chk("token_embd ne == [hidden_size, vocab_size]", [cfg["hidden_size"], V], tens.get("token_embd.weight"))
    tied = bool(cfg.get("tie_word_embeddings", False))
    chk("output.weight present == not tie_word_embeddings", not tied, "output.weight" in tens)
    if "bos_token_id" in cfg:
        chk("tokenizer.ggml.bos_token_id == bos_token_id", cfg["bos_token_id"], get("tokenizer.ggml.bos_token_id"))
    if "eos_token_id" in cfg:
        chk("tokenizer.ggml.eos_token_id == eos_token_id", cfg["eos_token_id"], get("tokenizer.ggml.eos_token_id"))

    # weight shapes implied by the config (ggml ne order), every layer
    nh, nkv, hd = attn["num_heads"], attn["num_kv_heads"], attn["head_dim"]
    H = cfg["hidden_size"]
    bad = []
    for name, ne in tens.items():
        m = re.match(r"^blk\.(\d+)\.(.+)$", name)
        if not m:
            continue
        lt = recipes.get(name, {}).get("layer_type")
        sfx = m.group(2)
        want = None
        if lt == "nsa":
            want = {"attn_q.weight": [H, nh * hd], "attn_k.weight": [H, nkv * hd], "attn_v.weight": [H, nkv * hd],
                    "attn_output.weight": [nh * hd, H], "attn_gate.weight": [H, 3 * nh]}.get(sfx)
        elif lt == "gdn":
            want = {"attn_qkv.weight": [H, 2 * n_k * hk + n_v * hv], "ssm_conv1d.weight": [cfg["conv_size"], 2 * n_k * hk + n_v * hv],
                    "attn_gate.weight": [H, n_v * hv], "ssm_beta.weight": [H, n_v], "ssm_alpha.weight": [H, n_v],
                    "ssm_dt.bias": [n_v], "ssm_a": [n_v], "ssm_norm.weight": [hv], "ssm_out.weight": [n_v * hv, H]}.get(sfx)
        if sfx in ("ffn_gate.weight", "ffn_up.weight"):
            want = [H, n_ff]
        elif sfx == "ffn_down.weight":
            want = [n_ff, H]
        elif sfx in ("attn_norm.weight", "ffn_norm.weight"):
            want = [H]
        if want is not None and want != ne:
            bad.append({"name": name, "ne": ne, "expected": want})
    rows.append({"check": "per-layer weight ne implied by config", "expected": [], "actual": bad, "pass": not bad})
    return rows


# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir (config.json + *.safetensors)")
    ap.add_argument("--gguf", required=True, help="converted GGUF file")
    ap.add_argument("--out", default="l0_weights.json", help="output JSON")
    ap.add_argument("--expect-n-tensors", type=int, default=290, help="expected #GGUF tensors (0: skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-vec", type=int, default=4, help="random vectors per projection test")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    hf = load_hf(ckpt)
    reader = GGUFReader(args.gguf)
    recipes, skipped = build_recipes(cfg, set(hf))

    tensors = check_tensors(reader, recipes, hf)
    coverage = check_coverage(reader, recipes, skipped, hf, args.expect_n_tensors)
    projection = check_projection(args.gguf, reader, recipes, hf, args.seed, args.n_vec)
    metadata = check_metadata(reader, cfg, hf, recipes)

    def pf(rows):
        n = sum(1 for r in rows if r["pass"])
        return {"pass": n, "fail": len(rows) - n}

    summary = {
        "tensors_bitwise": pf(tensors),
        "coverage": {"pass": sum(coverage["checks"].values()), "fail": sum(not v for v in coverage["checks"].values())},
        "projection": pf(projection),
        "metadata": pf(metadata),
    }
    ok = all(v["fail"] == 0 for v in summary.values())

    # ---------------- print ----------------
    print(f"gguf: {args.gguf}\nckpt: {args.ckpt}\n")
    groups: OrderedDict[str, list] = OrderedDict()
    for r in tensors:
        groups.setdefault(group_key(r["name"], r.get("layer_type") or "global"), []).append(r)
    print("[1] tensors: rebuilt from HF, bitwise compare")
    print(f"  {'group':<34} {'n':>3} {'dtype':>5} {'ne':<14} {'bitwise':>8} {'max_abs':>8}")
    for g, rs in groups.items():
        nb = sum(r["bitwise_equal"] for r in rs)
        mad = max((r["max_abs_diff"] for r in rs if r["max_abs_diff"] is not None), default=None)
        nes = {tuple(r["ne"]) for r in rs}
        ne_s = str(list(next(iter(nes)))) if len(nes) == 1 else "mixed"
        print(f"  {g:<34} {len(rs):>3} {rs[0]['dtype']:>5} {ne_s:<14} {f'{nb}/{len(rs)}':>8} {fmt_e(mad):>8}")
    for r in tensors:
        if not r["pass"]:
            print(f"  FAIL {r['name']}: {r.get('error', '')} max_abs={r['max_abs_diff']} max_ulp={r['max_ulp']} "
                  f"n_diff={r.get('n_diff')}")

    c = coverage
    print("\n[2] coverage")
    print(f"  HF tensors {c['n_hf_tensors']}, used exactly once {c['n_hf_used_once']}, skipped {c['hf_skipped']}")
    print(f"  GGUF tensors {c['n_gguf_tensors']} (tensor_count field {c['gguf_tensor_count_field']}), "
          f"recipes {c['n_recipes']}, accounted {c['n_gguf_accounted']}")
    for name, v in c["checks"].items():
        print(f"  {'ok  ' if v else 'FAIL'} {name}")
    for key in ("hf_unused", "hf_used_more_than_once", "hf_missing_for_recipe", "gguf_without_recipe",
                "recipe_missing_in_gguf", "gguf_duplicate_names"):
        if c[key]:
            print(f"  {key}: {c[key][:10]}{' ...' if len(c[key]) > 10 else ''}")

    print("\n[3] projection: ggml-strided W from raw bytes, y = W x vs float64 HF reference")
    pg: OrderedDict[str, list] = OrderedDict()
    for r in projection:
        pg.setdefault(group_key(r["name"], r["layer_type"]), []).append(r)
    print(f"  {'group':<34} {'n':>3} {'ne0==in':>8} {'pass':>6} {'max_rel_err':>11} {'neg_ctrl_min':>12}")
    for g, rs in pg.items():
        n_in = sum(r["ne0_eq_in"] for r in rs)
        npass = sum(r["pass"] for r in rs)
        mre = max((r["max_rel_err"] for r in rs if r["max_rel_err"] is not None), default=None)
        ncs = [r["neg_control_rel_err"] for r in rs if r["neg_control_rel_err"] is not None]
        print(f"  {g:<34} {len(rs):>3} {f'{n_in}/{len(rs)}':>8} {f'{npass}/{len(rs)}':>6} {fmt_e(mre):>11} "
              f"{fmt_e(min(ncs)) if ncs else '-':>12}")
    for r in projection:
        if "get_rows_bitwise_equal" in r:
            print(f"  token_embd get_rows (64 random ids) bitwise equal: {r['get_rows_bitwise_equal']}")
        if not r["pass"]:
            print(f"  FAIL {r['name']}: {r.get('error', '')} rel_err={r['max_rel_err']} neg={r['neg_control_rel_err']}")

    print("\n[4] metadata vs config.json")
    for r in metadata:
        if r["check"].startswith("per-layer"):
            print(f"  {'ok  ' if r['pass'] else 'FAIL'} {r['check']}" + ("" if r["pass"] else f": {r['actual'][:5]}"))
            continue
        a = r["actual"]
        a_s = (f"[{len(a)} items]" if isinstance(a, list) and len(a) > 4 else repr(a))
        e = r["expected"]
        e_s = (f"[{len(e)} items]" if isinstance(e, list) and len(e) > 4 else repr(e))
        print(f"  {'ok  ' if r['pass'] else 'FAIL'} {r['check']:<62} exp={e_s} got={a_s}")

    print("\nsummary")
    for name, v in summary.items():
        print(f"  {name:<16} pass {v['pass']:>4}  fail {v['fail']:>3}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "gguf": args.gguf, "ckpt": args.ckpt, "ok": ok, "summary": summary,
        "tolerances": {"projection_rel": PROJ_REL_TOL, "neg_control_min": NEG_CONTROL_MIN},
        "coverage": coverage, "metadata": metadata, "tensors": tensors, "projection": projection,
    }
    out.write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
