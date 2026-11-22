#!/usr/bin/env bash
# L0-L5 correctness suite of the gdn-nsa port (tests/gdn-nsa-test/REPORT.md). Runs the level-specific checks and writes
# one report with PASS/FAIL per check. The model-level dumps (L2/L3/L5) come from run_validation.sh (MODEL_RESULTS).
#   MODEL=<f32 gguf> GOLDEN=<golden dir> CKPT=<HF checkpoint dir> REFREPO=<fla repo> PY=<reference venv python>
#   MODEL_RESULTS=<run_validation.sh results dir> tests/gdn-nsa-test/scripts/run_levels.sh [GPU index] [results dir]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${1:-0}
R=${2:-results-levels}
M=${MODEL:?set MODEL}; G=${GOLDEN:?set GOLDEN}; C=${CKPT:?set CKPT}; RR=${REFREPO:?set REFREPO}; PY=${PY:?set PY}
R3=${MODEL_RESULTS:?set MODEL_RESULTS (run_validation.sh output)}
BIN=${BIN:-$ROOT/build-gdn-nsa}
mkdir -p "$R"
# the checkers exit 1 on a FAIL; keep going so the report shows it, and drop old JSONs so a crash reads "not run"
rm -f "$R/l0_weights.json" "$R/l1_gdn.json" "$R/l1_nsa.json" "$R/l1_nsa_notf32.json" "$R"/l34_*_len600.json
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')

# build every C++ tool (against build-cpu/ and build-cuda/)
bash "$ROOT/tests/gdn-nsa-test/scripts/build.sh" "$BIN"

# L0: GGUF tensors are an exact conversion of the checkpoint
PYTHONPATH="$ROOT/gguf-py" $PY "$ROOT/tests/gdn-nsa-test/analysis/check_gguf_weights.py" --ckpt "$C" --gguf "$M" --out "$R/l0_weights.json" \
    > "$R/l0_weights.txt" || echo "L0 check_gguf_weights.py exit $?" >&2

# L1: GDN op vs fp64 oracle (+ prefill vs token-by-token state carry); NSA ops vs fla fixtures
"$BIN/gdn-oracle-test" "$R/l1_gdn.json" > "$R/l1_gdn.txt" || echo "L1 gdn-oracle-test exit $?" >&2
FIX=${FIXTURES:-fixtures/nsa}   # written by make_nsa_fixtures.py on first use (fla naive path, fp32/f64; 351 MB)
[[ -f $FIX/cases.txt ]] || $PY "$ROOT/tests/gdn-nsa-test/reference/make_nsa_fixtures.py" "$FIX" > "$R/l1_nsa_fixtures.txt"
"$BIN/nsa-fixture-test" "$FIX" "$R/l1_nsa.json" > "$R/l1_nsa.txt" || echo "L1 nsa-fixture-test exit $?" >&2
NVIDIA_TF32_OVERRIDE=0 "$BIN/nsa-fixture-test" "$FIX" "$R/l1_nsa_notf32.json" > "$R/l1_nsa_notf32.txt" \
    || echo "L1 nsa-fixture-test (TF32 off) exit $?" >&2   # strict F32, TF32 off

# L3/L4: prefill at block/window boundary lengths, prefill(S) + token-by-token decode splits
for be in cuda cpu; do
  ngl=0; [[ $be == cuda ]] && ngl=99
  "$BIN/nsa-cache-equiv-$be" -m "$M" -t "$G/tokens_len600.txt" -ngl $ngl --ref "$G/logits_len600.npy" \
      --ref32 "$G/logits_len600_fp32.npy" --json "$R/l34_${be}_len600.json" > "$R/l34_${be}_len600.txt" 2> "$R/l34_${be}_len600.log"
done

# report (+ reference_manifest.json; GPU picks the nvidia-smi row it records)
GPU=$GPU $PY "$ROOT/tests/gdn-nsa-test/analysis/levels_report.py" --golden "$G" --model-results "$R3" --results "$R" --ckpt "$C" --gguf "$M" \
    --ref-repo "$RR" --out "$R/report.md"
