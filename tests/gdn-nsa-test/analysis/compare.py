#!/usr/bin/env python3
"""Compare llama.cpp gdn-nsa logits / block selections against the PyTorch golden outputs.

usage: compare.py --ref logits_NAME.npy --ours out.logits.bin [--ref-idx blockidx_NAME.npy --ours-idx out.idx.bin]
                  [--margin margin_NAME.npy] [--json out.json] [--label text]
"""
import argparse
import json

import numpy as np


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def logits_metrics(ref, ours):
    d = np.abs(ref - ours)
    lp, lq = log_softmax(ref.astype(np.float64)), log_softmax(ours.astype(np.float64))
    kl = (np.exp(lp) * (lp - lq)).sum(-1)  # KL(ref || ours) per position
    top1 = ref.argmax(-1) == ours.argmax(-1)
    T = ref.shape[0]
    out = {
        "T": int(T),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "top1_agree": float(top1.mean()),
        "argmax_pos_max_abs": int(d.max(-1).argmax()),
    }
    bands = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    for a, b in bands:
        if a < T:
            sl = slice(a, min(b, T))
            out[f"band_{a}_{b}"] = {
                "max_abs": float(d[sl].max()),
                "kl_mean": float(kl[sl].mean()),
                "top1": float(top1[sl].mean()),
            }
    return out


def idx_metrics(ref_idx, ours_idx, margin=None):
    # ref_idx: [L, T, H, S]; ours_idx flat [L, T, H, S]
    L, T, H, S = ref_idx.shape
    ours_idx = ours_idx.reshape(L, T, H, S)
    agree, exact, n = 0.0, 0, 0
    mism = []
    for l in range(L):
        for t in range(T):
            for h in range(H):
                a = set(int(x) for x in ref_idx[l, t, h] if x >= 0)
                b = set(int(x) for x in ours_idx[l, t, h] if x >= 0)
                n += 1
                inter = len(a & b)
                agree += inter / max(1, len(a))
                if a == b:
                    exact += 1
                else:
                    mism.append((l, t, h, len(a ^ b) // 2, float(margin[l, t, h]) if margin is not None else None))
    out = {"sets": n, "exact_set_agree": exact / n, "mean_overlap": agree / n, "n_mismatch": len(mism)}
    if margin is not None and mism:
        m = np.array([x[4] for x in mism])
        out["mismatch_margin_max"] = float(np.abs(m).max())
        out["mismatch_margin_median"] = float(np.median(np.abs(m)))
        out["all_margin_median"] = float(np.median(np.abs(margin)))
    out["first_mismatches"] = mism[:10]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref-idx")
    ap.add_argument("--ours-idx")
    ap.add_argument("--margin")
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ref = np.load(a.ref).astype(np.float32)
    ours = np.fromfile(a.ours, dtype=np.float32).reshape(ref.shape[0], -1)
    res = {"label": a.label, "logits": logits_metrics(ref, ours)}
    if a.ref_idx and a.ours_idx:
        ref_idx = np.load(a.ref_idx)
        ours_idx = np.fromfile(a.ours_idx, dtype=np.int32)
        margin = np.load(a.margin) if a.margin else None
        res["blocks"] = idx_metrics(ref_idx, ours_idx, margin)
    s = json.dumps(res, indent=1)
    print(s)
    if a.json:
        open(a.json, "w").write(s)


if __name__ == "__main__":
    main()
