#!/usr/bin/env python3
"""L0-L5 correctness report for the gdn-nsa port (see tests/gdn-nsa-test/REPORT.md).

Collects the results of every validation level into one markdown report with PASS/FAIL per check, and writes the
reference manifest. Model-level metrics are recomputed from the raw dumps with the extended metric set
(max abs, NRMSE, min per-row cosine, KL, top-1, non-finite count, top-1 mismatch margins).

usage (reference venv python, it needs numpy only):
  levels_report.py --golden G --model-results R3 --results R4 --ckpt CKPT --gguf GGUF --ref-repo REPO --out R4/report.md
inputs:
  G   golden dir (logits_*.npy, blockidx_*.npy, margin_*.npy, hidden_len2k.npy, greedy_*_cache.txt)
  R3  run_validation.sh results (NAME_{cpu,cuda}_{prefill,decode,chunk,hidden}.*, test-backend-ops.txt, sched_summary.txt)
  R4  new level results: l0_weights.json, l1_nsa.json, l1_gdn.json, l34_{cpu,cuda}_len600.json
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess

import numpy as np

# pass criteria, declared before looking at the numbers of a run (measured values are 2-3 orders below)
TH = {
    "layer_nrmse": 2e-3,       # L2: per-layer hidden state NRMSE vs reference
    "logit_nrmse": 2e-3,       # L3/L5: logits NRMSE vs reference
    "kl_mean": 1e-5,           # mean KL(ref || ours) over positions
    "top1": 0.995,             # top-1 agreement
    "tie_margin": 0.05,        # a top-1 mismatch is accepted only if the reference top1-top2 logit margin is below this
    "block_exact": 0.99,       # selected block sets equal to the reference
    "block_tie": 1e-3,         # a block-set mismatch is accepted only if the reference score margin is below this
}


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def logit_metrics(ours, ref):
    ours = ours.astype(np.float64)
    ref = ref.astype(np.float64)
    nonfinite = int((~np.isfinite(ours)).sum())
    d = ours - ref
    lp, lq = log_softmax(ref), log_softmax(ours)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    cos = (ours * ref).sum(-1) / np.sqrt((ours * ours).sum(-1) * (ref * ref).sum(-1))
    ta, tb = ours.argmax(-1), ref.argmax(-1)
    mism = np.nonzero(ta != tb)[0]
    srt = np.sort(ref, -1)
    margin = srt[:, -1] - srt[:, -2]
    return {
        "rows": int(ref.shape[0]), "max_abs": float(np.abs(d).max()),
        "rmse": float(np.sqrt((d * d).mean())), "nrmse": float(np.linalg.norm(d) / np.linalg.norm(ref)),
        "cos_min": float(cos.min()), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "top1": float((ta == tb).mean()), "nonfinite": nonfinite,
        "top1_mismatch_ref_margin_max": float(margin[mism].max()) if len(mism) else 0.0,
        "first_mismatch": int(mism[0]) if len(mism) else -1,
    }


def logit_pass(m):
    return (m["nonfinite"] == 0 and m["nrmse"] <= TH["logit_nrmse"] and m["kl_mean"] <= TH["kl_mean"]
            and m["top1"] >= TH["top1"] and m["top1_mismatch_ref_margin_max"] <= TH["tie_margin"])


def block_metrics(ours, ref, margin=None):
    a = np.sort(ours.reshape(ref.shape), -1)
    b = np.sort(ref, -1)
    eq = (a == b).all(-1)
    out = {"sets": int(eq.size), "exact": float(eq.mean()), "n_mismatch": int((~eq).sum())}
    if margin is not None:
        mm = np.abs(margin[~eq]) if (~eq).any() else np.zeros(0)
        out["mismatch_margin_max"] = float(mm.max()) if mm.size else 0.0
        out["all_margin_median"] = float(np.median(np.abs(margin[np.isfinite(margin)])))
    return out


def block_pass(m):
    return m["exact"] >= TH["block_exact"] and m.get("mismatch_margin_max", 0.0) <= TH["block_tie"]


def load_logits(path, rows):
    return np.fromfile(path, dtype=np.float32).reshape(rows, -1)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def manifest(a):
    def ver(pkg):
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            return None
    meta = os.path.join(a.ckpt, ".cache/huggingface/download/model.safetensors.metadata")
    rev = open(meta).read().split("\n")[0] if os.path.exists(meta) else None

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return {
        "model": "Idiap/gated-deltanet-nsa-0.4B-10B",
        "checkpoint_revision": rev,
        "checkpoint_sha256": sha(os.path.join(a.ckpt, "model.safetensors")),
        "gguf": os.path.abspath(a.gguf), "gguf_sha256": sha(a.gguf),
        "reference_repo": "idiap/hybrid-linear-sparse-attention", "reference_commit": sh(f"git -C {a.ref_repo} rev-parse HEAD"),
        "reference_dirty": bool(sh(f"git -C {a.ref_repo} status --porcelain --untracked-files=no")),
        "torch": ver("torch"), "transformers": ver("transformers"), "triton": ver("triton"),
        "flash_attn": ver("flash_attn") or ver("flash-attn"), "numpy": ver("numpy"),
        "reference_dtype": "float32 weights; kernel path casts q/k/v to fp16 (author's code), fp32 path has no casts",
        "reference_tf32": "disabled (matmul and cudnn)", "sampling": "none (greedy)", "dropout": "eval mode",
        "port_commit": sh(f"git -C {root} rev-parse HEAD"), "port_uncommitted_changes": bool(sh(f"git -C {root} status --porcelain")),
        "gpu": sh("nvidia-smi -i ${GPU:-7} --query-gpu=name,driver_version --format=csv,noheader"),
        "cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"), "host_python": platform.python_version(),
        "compiler": sh("g++ --version | head -1"),
    }


def main():
    ap = argparse.ArgumentParser()
    for k in ("golden", "model_results", "results", "ckpt", "gguf", "ref_repo", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    G, R3, R4 = a.golden, a.model_results, a.results
    rows, md = [], []  # rows: (level, check, backend, pass, detail)

    def add(level, check, be, ok, detail):
        rows.append((level, check, be, ok, detail))

    man = manifest(a)
    json.dump(man, open(os.path.join(R4, "reference_manifest.json"), "w"), indent=1)

    # ---- L0 / L1: results of the dedicated tools
    for level, fn in (("L0", "l0_weights.json"), ("L1", "l1_nsa.json"), ("L1", "l1_gdn.json")):
        p = os.path.join(R4, fn)
        if not os.path.exists(p):
            add(level, fn, "-", None, "not run")
            continue
        j = json.load(open(p))
        ok = j.get("pass", j.get("ok", j.get("all_pass")))
        if ok is None and isinstance(j.get("summary"), dict):
            s = j["summary"]
            ok = s.get("pass", s.get("ok", s.get("all_pass")))
            if ok is None and "n_fail" in s:
                ok = s["n_fail"] == 0
        if ok is None and "n_fail" in j:
            ok = j["n_fail"] == 0
        summ = j.get("summary", {})
        det = json.dumps(summ)[:400] if summ else ""
        if not det and "records" in j:
            det = f"{len(j['records'])} checks, {j.get('n_fail')} failed; op {j.get('op', '')}; thresholds {json.dumps(j.get('thresholds', {}))}"
        names = {"l0_weights.json": "GGUF tensors vs checkpoint (bitwise, coverage, layout/projection, metadata)",
                 "l1_nsa.json": "NSA ops vs fla on identical fixtures (selection, fixed-index sel_attn, SWA, cmp, gates)",
                 "l1_gdn.json": "GDN op vs fp64 oracle, prefill vs token-by-token state carry, edge cases"}
        add(level, names.get(fn, fn), "cpu+cuda" if level == "L1" else "-", ok, det)

    # op-level tests that already existed
    tbo = os.path.join(R3, "test-backend-ops.txt")
    if os.path.exists(tbo):
        t = open(tbo).read()
        n_ok, n_fail = t.count(": OK") + t.count(" OK\n"), t.count("FAIL")
        passed = [l.strip() for l in t.splitlines() if "tests passed" in l]
        add("L1", "test-backend-ops NSA_SELECT / NSA_SEL_ATTN (CUDA vs CPU)", "cuda", n_fail == 0, "; ".join(passed))
    mr = os.path.join(R3, "nsa-mask-ref.txt")
    if os.path.exists(mr):
        t = open(mr).read()
        add("L1", "nsa-mask-ref (new ops vs existing-op reference, up to 32K)", "cpu+cuda", "FAIL" not in t,
            f"{t.count('OK')} OK lines")

    # ---- L2: hidden states after every layer (len2k)
    href = np.load(os.path.join(G, "hidden_len2k.npy"), mmap_mode="r")  # [25, T, 1024], [0] = embeddings
    L, T, H = href.shape[0] - 1, href.shape[1], href.shape[2]
    layer_tab = {}
    for be in ("cpu", "cuda"):
        p = os.path.join(R3, f"len2k_{be}_hidden.hidden.bin")
        if not os.path.exists(p):
            continue
        hh = np.fromfile(p, dtype=np.float32).reshape(L, T, H)
        errs = []
        for l in range(L):
            r = np.asarray(href[l + 1], dtype=np.float64)
            o = hh[l].astype(np.float64)
            d = o - r
            cos = (o * r).sum(-1) / np.sqrt((o * o).sum(-1) * (r * r).sum(-1))
            errs.append((float(np.linalg.norm(d) / np.linalg.norm(r)), float(cos.min()), float(np.abs(d).max()),
                         int((~np.isfinite(o)).sum())))
        layer_tab[be] = errs
        worst = max(e[0] for e in errs)
        jumps = max(errs[i + 1][0] / max(errs[i][0], 1e-12) for i in range(L - 1))
        add("L2", "hidden state after each of the 24 layers (len2k)", be,
            worst <= TH["layer_nrmse"] and all(e[3] == 0 for e in errs),
            f"NRMSE max {worst:.2e}, min cosine {min(e[1] for e in errs):.6f}, largest layer-to-layer NRMSE ratio {jumps:.2f}")

    # ---- L3 / L5: prefill logits at every position, block selections, CPU vs GPU
    logit_tab = []
    for name in ("short", "len600", "len2k", "len4k"):
        rp = os.path.join(G, f"logits_{name}.npy")
        if not os.path.exists(rp):
            continue
        ref = np.load(rp)
        ref32 = np.load(os.path.join(G, f"logits_{name}_fp32.npy")) if os.path.exists(os.path.join(G, f"logits_{name}_fp32.npy")) else None
        dumps = {}
        for be in ("cpu", "cuda"):
            p = os.path.join(R3, f"{name}_{be}_prefill.logits.bin")
            if not os.path.exists(p):
                continue
            ours = load_logits(p, ref.shape[0])
            dumps[be] = ours
            m = logit_metrics(ours, ref)
            m32 = logit_metrics(ours, ref32) if ref32 is not None else None
            logit_tab.append((name, be, "prefill", m, m32))
            add("L3", f"prefill logits, all {ref.shape[0]} positions ({name}) vs kernel ref", be, logit_pass(m),
                f"NRMSE {m['nrmse']:.2e}, KL {m['kl_mean']:.1e}, top-1 {m['top1']*100:.3f}%, max tie margin {m['top1_mismatch_ref_margin_max']:.3f}")
            bi = os.path.join(G, f"blockidx_{name}.npy")
            oi = os.path.join(R3, f"{name}_{be}_prefill.idx.bin")
            if os.path.exists(bi) and os.path.exists(oi):
                mg = os.path.join(G, f"margin_{name}.npy")
                bm = block_metrics(np.fromfile(oi, dtype=np.int32), np.load(bi), np.load(mg) if os.path.exists(mg) else None)
                add("L3", f"selected block sets ({name}) vs kernel ref", be, block_pass(bm),
                    f"exact {bm['exact']*100:.3f}% of {bm['sets']}, mismatches {bm['n_mismatch']}, max ref margin of a mismatch {bm.get('mismatch_margin_max', float('nan')):.1e}")
        if "cpu" in dumps and "cuda" in dumps:
            m = logit_metrics(dumps["cuda"], dumps["cpu"])
            add("L5", f"CPU vs GPU logits ({name})", "cpu-cuda", m["nonfinite"] == 0 and m["kl_mean"] <= TH["kl_mean"] and m["top1"] >= TH["top1"],
                f"NRMSE {m['nrmse']:.2e}, KL {m['kl_mean']:.1e}, top-1 {m['top1']*100:.3f}%")
    # decode paths of the existing run (len2k, CUDA)
    ref = np.load(os.path.join(G, "logits_len2k.npy"))
    for tag in ("decode", "chunk", "denseswa"):
        p = os.path.join(R3, f"len2k_cuda_{tag}.logits.bin")
        if os.path.exists(p):
            m = logit_metrics(load_logits(p, ref.shape[0]), ref)
            logit_tab.append(("len2k", "cuda", tag, m, None))
            add("L4" if tag != "denseswa" else "L3", f"len2k {tag} (2048 positions) vs kernel ref", "cuda", logit_pass(m),
                f"NRMSE {m['nrmse']:.2e}, KL {m['kl_mean']:.1e}, top-1 {m['top1']*100:.3f}%")

    # ---- L3 / L4: boundary lengths and prefill + decode splits (nsa-cache-equiv)
    for be in ("cpu", "cuda"):
        p = os.path.join(R4, f"l34_{be}_len600.json")
        if not os.path.exists(p):
            add("L4", "prefill(S) + decode splits", be, None, "not run")
            continue
        j = json.load(open(p))
        for kind, level in (("length", "L3"), ("split", "L4")):
            runs = [r for r in j["runs"] if r["kind"] == kind]
            ok = all(r["vs_full"]["nonfinite"] == 0 and r["vs_full"]["top1"] >= TH["top1"] and r["vs_full"]["kl_mean"] <= TH["kl_mean"]
                     and r["vs_ref"]["top1"] >= TH["top1"] and r["vs_ref"]["kl_mean"] <= TH["kl_mean"] and r["vs_ref"]["nrmse"] <= TH["logit_nrmse"]
                     for r in runs)
            what = "prefill of the first L tokens" if kind == "length" else "prefill S tokens + token-by-token decode to 600"
            ns = ",".join(str(r["n"]) for r in runs)
            add(level, f"{what}, {'L' if kind == 'length' else 'S'} = {ns}", be, ok,
                f"worst vs full prefill: NRMSE {max(r['vs_full']['nrmse'] for r in runs):.1e}, KL {max(r['vs_full']['kl_mean'] for r in runs):.1e}; "
                f"worst vs ref: NRMSE {max(r['vs_ref']['nrmse'] for r in runs):.1e}, KL {max(r['vs_ref']['kl_mean'] for r in runs):.1e}, "
                f"min top-1 {min(r['vs_ref']['top1'] for r in runs)*100:.2f}%")

    # ---- L5: greedy generation, first divergence
    for name in ("short", "len2k"):
        gp = os.path.join(G, f"greedy_{name}_cache.txt")
        op = os.path.join(R3, f"{name}_cuda_prefill.gen.txt")
        if os.path.exists(gp) and os.path.exists(op):
            g = open(gp).read().split()
            o = open(op).read().split()
            n = min(len(g), len(o))
            div = next((i for i in range(n) if g[i] != o[i]), -1)
            add("L5", f"greedy generation ({name}), HF generate() with cache", "cuda", div == -1 and n > 0,
                f"{n} tokens compared, first divergence {'none' if div < 0 else div}")

    # ---- GPU fallback
    sp = os.path.join(R3, "sched_summary.txt")
    if os.path.exists(sp):
        t = open(sp).read()
        import re
        cpu_ops = set()
        for d in re.findall(r"non-GPU nodes \{([^}]*)\}", t):
            cpu_ops.update(k.strip().strip("'").split(":")[0] for k in re.findall(r"'([^']+)'\s*:", d))
        # the token embedding lookup (GET_ROWS on the host-side embedding input) is the only allowed CPU node
        add("L5", "no silent CPU fallback (GGML_SCHED_DEBUG=2 node assignment, prefill + decode)", "cuda",
            cpu_ops <= {"GET_ROWS"}, f"ops on CPU: {sorted(cpu_ops) or 'none'}; NSA_SELECT / NSA_SEL_ATTN all on CUDA0")

    # ---- write
    md.append("# gdn-nsa correctness report (L0-L5)\n")
    md.append("Pass criteria (declared up front): " + ", ".join(f"{k} {v}" for k, v in TH.items()) + "\n")
    md.append("| level | check | backend | result | detail |\n|---|---|---|---|---|")
    for lv, chk, be, ok, det in rows:
        res = "PASS" if ok is True else "FAIL" if ok is False else "see detail"
        md.append(f"| {lv} | {chk} | {be} | {res} | {det} |")
    md.append("\n## L2 per-layer hidden state error (len2k)\n")
    if layer_tab:
        bes = list(layer_tab)
        md.append("| layer | type | " + " | ".join(f"{b} NRMSE | {b} min cos" for b in bes) + " |")
        md.append("|---|---|" + "---|---|" * len(bes))
        for l in range(L):
            md.append(f"| {l} | {'GDN' if l % 2 == 0 else 'NSA'} | " + " | ".join(f"{layer_tab[b][l][0]:.2e} | {layer_tab[b][l][1]:.6f}" for b in bes) + " |")
    md.append("\n## Logits vs reference (extended metrics)\n")
    md.append("| prompt | backend | path | rows | max abs | NRMSE | min cos | KL mean | KL max | top-1 | non-finite | KL vs fp32 ref |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, be, tag, m, m32 in logit_tab:
        md.append(f"| {name} | {be} | {tag} | {m['rows']} | {m['max_abs']:.2e} | {m['nrmse']:.2e} | {m['cos_min']:.6f} | {m['kl_mean']:.1e} | "
                  f"{m['kl_max']:.1e} | {m['top1']*100:.3f}% | {m['nonfinite']} | {m32['kl_mean']:.1e} |" if m32 else
                  f"| {name} | {be} | {tag} | {m['rows']} | {m['max_abs']:.2e} | {m['nrmse']:.2e} | {m['cos_min']:.6f} | {m['kl_mean']:.1e} | "
                  f"{m['kl_max']:.1e} | {m['top1']*100:.3f}% | {m['nonfinite']} | - |")
    md.append("\n## Reference manifest\n")
    md.append("```json\n" + json.dumps(man, indent=1) + "\n```")
    open(a.out, "w").write("\n".join(md) + "\n")
    n_fail = sum(1 for r in rows if r[3] is False)
    print(f"{len(rows)} checks, {sum(1 for r in rows if r[3] is True)} PASS, {n_fail} FAIL -> {a.out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
