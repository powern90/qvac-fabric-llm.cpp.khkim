#!/usr/bin/env python3
"""Markdown tables for the quantized KV cache runs (run_validation.sh step 6).

usage: kvq_summary.py <results dir> <golden dir>

Run names: NAME_BE_KV[_VARIANT] (NAME = len600/len2k/len4k, BE = cuda/cpu, KV = q8_0/q4_0/q8_0_q4_0/...,
VARIANT = decode/chunk/rot/rot_decode/denseswa), plus the F16 baselines NAME_BE_f16[_VARIANT].
References in the golden dir:
  logits_NAME_fp32.npy             pure fp32 PyTorch (F16 KV is not modelled)
  logits_NAME_fp32_kvq8_0.npy ...  the same with K/V quantized + dequantized exactly as ggml (fake quant), if present
"""
import glob
import os
import sys

import numpy as np

BS, S = 64, 16


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl_top1(ref, ours):
    lp, lq = log_softmax(ref.astype(np.float64)), log_softmax(ours.astype(np.float64))
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    return float(kl.mean()), float(kl.max()), float((ref.argmax(-1) == ours.argmax(-1)).mean())


def ppl(lg, toks):
    lp = log_softmax(lg[:-1].astype(np.float64))
    return float(np.exp(-lp[np.arange(len(toks) - 1), toks[1:]].mean()))


def sel_overlap(a, b):
    """mean set overlap and exact-set rate over the positions with more complete blocks than selected ones"""
    t0 = (S + 1) * BS
    a, b = a[:, t0:], b[:, t0:]
    if a.shape[1] == 0:
        return float("nan"), float("nan")
    inter = np.zeros(a.shape[:3])
    for i in range(a.shape[-1]):
        inter += ((a[..., i:i + 1] == b) & (a[..., i:i + 1] >= 0)).any(-1)
    ov = inter / np.maximum(1, (a >= 0).sum(-1))
    return float(ov.mean()), float((ov == 1).mean())


def main():
    R, G = sys.argv[1], sys.argv[2]
    cache = {}

    def logits(run, T):
        if run not in cache:
            cache[run] = np.fromfile(os.path.join(R, run + ".logits.bin"), dtype=np.float32).reshape(T, -1)
        return cache[run]

    def idx(run, L, T, H):
        p = os.path.join(R, run + ".idx.bin")
        return np.fromfile(p, dtype=np.int32).reshape(L, T, H, S) if os.path.exists(p) else None

    runs = sorted(os.path.basename(f)[:-len(".logits.bin")] for f in glob.glob(os.path.join(R, "*.logits.bin")))
    runs = [r for r in runs if r.split("_")[0] in ("len600", "len2k", "len4k") and "_sched_" not in r]

    print("| run | ppl | Δppl vs fp32 ref | KL vs fp32 ref | top-1 vs fp32 ref | KL vs fake-quant ref | top-1 vs fake-quant ref "
          "| block overlap vs F16 run | exact block sets vs F16 run |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        name, be, rest = run.split("_", 2)
        tok_path, ref_path = os.path.join(G, f"tokens_{name}.txt"), os.path.join(G, f"logits_{name}_fp32.npy")
        if not (os.path.exists(tok_path) and os.path.exists(ref_path)):
            print(f"| {run} | (no tokens_{name}.txt / logits_{name}_fp32.npy) | | | | | | | |")
            continue
        toks = np.array(open(tok_path).read().split(), dtype=np.int64)
        ref = np.load(ref_path).astype(np.float32)
        T = ref.shape[0]
        ours = logits(run, T)
        p_ref, p = ppl(ref, toks), ppl(ours, toks)
        kl, _, top1 = kl_top1(ref, ours)

        # fake-quant reference: only for runs with the same K and V type and no rotation
        kv = rest
        for v in ("_decode", "_chunk", "_denseswa", "_rot_decode", "_rot"):
            if kv.endswith(v):
                kv = kv[:-len(v)]
        fq = os.path.join(G, f"logits_{name}_fp32_kv{kv}.npy")
        fq_cols = ["-", "-"]
        if os.path.exists(fq) and "_rot" not in rest:
            k2, _, t2 = kl_top1(np.load(fq).astype(np.float32), ours)
            fq_cols = [f"{k2:.2e}", f"{100 * t2:.2f}%"]

        # block selection vs the F16 cache run of the same name/backend (prefill)
        sel_cols = ["-", "-"]
        base = f"{name}_{be}_f16"
        ref_idx = os.path.join(G, f"blockidx_{name}.npy")
        if os.path.exists(ref_idx) and os.path.exists(os.path.join(R, base + ".idx.bin")):
            L, _, H, _ = np.load(ref_idx).shape
            a, b = idx(base, L, T, H), idx(run, L, T, H)
            if a is not None and b is not None and run != base:
                ov, ex = sel_overlap(a, b)
                if ov == ov:  # nan: no position with more complete blocks than selected ones (T <= 1088)
                    sel_cols = [f"{100 * ov:.2f}%", f"{100 * ex:.2f}%"]

        print(f"| {run} | {p:.4f} | {100 * (p / p_ref - 1):+.2f}% | {kl:.2e} | {100 * top1:.2f}% | {fq_cols[0]} | {fq_cols[1]} "
              f"| {sel_cols[0]} | {sel_cols[1]} |")

    # consistency of the execution paths on the same quantized cache
    print()
    print("| pair | KL | top-1 | max abs |")
    print("|---|---|---|---|")
    pairs = []
    for run in runs:
        name, be, rest = run.split("_", 2)
        for v in ("_decode", "_chunk", "_denseswa"):
            if rest.endswith(v) and not rest.endswith("_rot_decode"):
                pairs.append((f"{name}_{be}_{rest[:-len(v)]}", run))
        if rest.endswith("_rot_decode"):
            pairs.append((f"{name}_{be}_{rest[:-len('_decode')]}", run))
        if be == "cpu":
            pairs.append((f"{name}_cuda_{rest}", run))
    for a, b in pairs:
        if not os.path.exists(os.path.join(R, a + ".logits.bin")):
            continue
        name = a.split("_")[0]
        tok_path = os.path.join(G, f"tokens_{name}.txt")
        if not os.path.exists(tok_path):
            continue
        T = len(open(tok_path).read().split())
        la, lb = logits(a, T), logits(b, T)
        kl, _, top1 = kl_top1(la, lb)
        print(f"| {a} vs {b} | {kl:.2e} | {100 * top1:.2f}% | {np.abs(la - lb).max():.2e} |")


if __name__ == "__main__":
    main()
