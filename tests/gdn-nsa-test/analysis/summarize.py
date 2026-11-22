#!/usr/bin/env python3
"""Prints markdown tables from the run_validation.sh results.

usage: summarize.py <results dir> <golden dir>
"""
import glob
import json
import os
import sys

import numpy as np


def main():
    R, G = sys.argv[1], sys.argv[2]

    print("| run | vs | T | max abs | mean abs | KL mean | KL max | top-1 | block sets equal | mismatches | max margin of a mismatch |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(R, "cmp_*.json"))):
        r = json.load(open(f))
        tag = os.path.basename(f)[4:-5]
        vs = "fp32" if tag.endswith("_fp32") else "kernel"
        tag = tag[:-5] if vs == "fp32" else tag
        l, b = r["logits"], r.get("blocks")
        blk = ("%.3f%%" % (100 * b["exact_set_agree"]), str(b["n_mismatch"]),
               "%.1e" % b["mismatch_margin_max"] if b.get("mismatch_margin_max") is not None else "-") if b else ("", "", "")
        print("| %s | %s | %d | %.2e | %.2e | %.1e | %.1e | %.3f%% | %s | %s | %s |" % (
            tag, vs, l["T"], l["max_abs"], l["mean_abs"], l["kl_mean"], l["kl_max"], 100 * l["top1_agree"], *blk))

    # greedy continuations
    print()
    for name in ["short", "len2k"]:
        ours = os.path.join(R, f"{name}_cuda_prefill.gen.txt")
        if not os.path.exists(ours):
            continue
        b = open(ours).read().split()
        for kind in ["nocache", "cache"]:
            ref = os.path.join(G, f"greedy_{name}_{kind}.txt")
            if os.path.exists(ref):
                a = open(ref).read().split()
                n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"- greedy {name}: llama.cpp CUDA vs reference ({kind}): first {n}/{min(len(a), len(b))} tokens identical")

    # prefill vs decode / chunked (self consistency)
    T = len(open(os.path.join(G, "tokens_len2k.txt")).read().split())
    base = os.path.join(R, "len2k_cuda_prefill.logits.bin")
    if os.path.exists(base):
        a = np.fromfile(base, dtype=np.float32).reshape(T, -1)
        for tag in ["cuda_decode", "cuda_chunk", "cuda_denseswa"]:
            p = os.path.join(R, f"len2k_{tag}.logits.bin")
            if os.path.exists(p):
                c = np.fromfile(p, dtype=np.float32).reshape(T, -1)
                print(f"- len2k {tag} vs cuda prefill: max abs {np.abs(a - c).max():.2e}, top-1 {(a.argmax(-1) == c.argmax(-1)).mean() * 100:.2f}%")

    # perplexity over the same tokens
    def ppl(lg, toks):
        lg = lg[:-1].astype(np.float64)
        t = np.array(toks[1:])
        m = lg.max(-1, keepdims=True)
        lse = np.log(np.exp(lg - m).sum(-1)) + m[:, 0]
        return float(np.exp((lse - lg[np.arange(len(t)), t]).mean()))

    print()
    for name in ["len2k", "len4k"]:
        tp = os.path.join(G, f"tokens_{name}.txt")
        if not os.path.exists(tp):
            continue
        toks = [int(x) for x in open(tp).read().split()]
        cols = []
        for lab, path in [("ref kernel", f"{G}/logits_{name}.npy"), ("ref fp32", f"{G}/logits_{name}_fp32.npy"),
                          ("llama.cpp CPU", f"{R}/{name}_cpu_prefill.logits.bin"), ("llama.cpp CUDA", f"{R}/{name}_cuda_prefill.logits.bin")]:
            if not os.path.exists(path):
                continue
            arr = np.load(path) if path.endswith(".npy") else np.fromfile(path, dtype=np.float32).reshape(len(toks), -1)
            cols.append(f"{lab} {ppl(arr, toks):.4f}")
        print(f"- perplexity {name}: " + ", ".join(cols))

    # layer level
    hp = os.path.join(G, "hidden_len2k.npy")
    if os.path.exists(hp):
        ref = np.load(hp)
        print()
        print("| layer | type | rel. err CPU | rel. err CUDA |")
        print("|---|---|---|---|")
        hs = {}
        for be in ["cpu", "cuda"]:
            p = os.path.join(R, f"len2k_{be}_hidden.hidden.bin")
            if os.path.exists(p):
                hs[be] = np.fromfile(p, dtype=np.float32).reshape(ref.shape[0] - 1, ref.shape[1], ref.shape[2])
        for l in range(ref.shape[0] - 1):
            row = [f"{l}", "NSA" if l % 2 else "GDN"]
            for be in ["cpu", "cuda"]:
                if be in hs:
                    d = hs[be][l] - ref[l + 1]
                    row.append("%.1e" % (np.linalg.norm(d) / np.linalg.norm(ref[l + 1])))
                else:
                    row.append("-")
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
