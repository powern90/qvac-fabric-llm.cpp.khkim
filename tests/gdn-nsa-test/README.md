# gdn-nsa validation suite

Everything used to validate the gdn-nsa port against the FLA reference: the final report, the validation code, the
run scripts and the results of the final run.

- [REPORT.md](REPORT.md): final validation report (L0–L5 + RULER S-NIAH subset)
- overview, performance, limitations and the full copy-paste command list: the repository [README.md](../../README.md)

## Layout

```
tests/gdn-nsa-test/
├── README.md, REPORT.md
├── scripts/      entry points
│   ├── build.sh            build every C++ tool into build-gdn-nsa/ (against build-cpu/, build-cuda/)
│   ├── run_validation.sh   model level: dumps vs golden, op tests, fallback check, quantized KV      (L1-L5)
│   └── run_levels.sh       level checks + report.md (PASS/FAIL per check) + reference_manifest.json (L0-L4)
├── cpp/          C++ tools (ggml / llama API)
│   ├── nsa-dump.cpp           logits / block indices / hidden states / greedy dump               (L2, L3, L5)
│   ├── nsa-cache-equiv.cpp    boundary-length prefill, prefill(S) + token-by-token decode        (L3, L4)
│   ├── nsa-mask-ref.cpp       new NSA ops vs a reimplementation with existing ggml ops           (L1)
│   ├── nsa-fixture-test.cpp   NSA ops vs fla on identical tensors (fixtures from reference/)     (L1)
│   ├── gdn-oracle-test.cpp    GDN op vs fp64 oracle, prefill vs decode state carry              (L1)
│   └── nsa-generate.cpp       greedy generation for the RULER subset                             (functional)
├── analysis/     Python (numpy only)
│   ├── check_gguf_weights.py  GGUF vs HF checkpoint: bitwise, coverage, layout, metadata         (L0)
│   ├── compare.py             logits / block-set metrics of one dump vs the golden outputs
│   ├── summarize.py, sched_summary.py, kvq_summary.py   tables for run_validation.sh
│   └── levels_report.py       combined L0-L5 report with the extended metrics
├── reference/    PyTorch / fla reference (separate venv): golden outputs, NSA fixtures, op checks
├── bench/        RULER S-NIAH subset: sample generator, FLA runner, scorer
└── results/      final run (levels/, model/, bench/)
```

Large inputs stay outside the repository: the checkpoint, the GGUF, the golden outputs (`logits_*.npy`, …) and the
NSA fixtures (351 MB). How to produce them is in [REPORT.md § 10](REPORT.md#10-reproduce).

## Run

```bash
tests/gdn-nsa-test/scripts/build.sh
MODEL=<f32 gguf> GOLDEN=<golden dir> PY=<reference venv python> \
  tests/gdn-nsa-test/scripts/run_validation.sh <gpu> results
MODEL=<f32 gguf> GOLDEN=<golden dir> CKPT=<HF checkpoint> REFREPO=<fla repo> PY=<reference venv python> MODEL_RESULTS=results \
  tests/gdn-nsa-test/scripts/run_levels.sh <gpu> results-levels                   # -> results-levels/report.md
```
