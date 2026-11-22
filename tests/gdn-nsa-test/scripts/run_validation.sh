#!/usr/bin/env bash
# End-to-end validation of the gdn-nsa port against the PyTorch golden outputs (see tests/gdn-nsa-test/REPORT.md).
#   MODEL=<f32 gguf> GOLDEN=<golden dir> PY=<python with numpy> [BIN=<tools dir>] [CUDA_BUILD=<cuda build dir>]
#   tests/gdn-nsa-test/scripts/run_validation.sh [GPU index] [results dir]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${1:-0}
R=${2:-results}
M=${MODEL:?set MODEL to the F32 GGUF}
G=${GOLDEN:?set GOLDEN to the golden dir}
PY=${PY:-python3}
BIN=${BIN:-$ROOT/build-gdn-nsa}
CUDA_BUILD=${CUDA_BUILD:-$ROOT/build-cuda}
CMP="$PY -I $ROOT/tests/gdn-nsa-test/analysis/compare.py"
mkdir -p "$R"
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')

run() { # name backend tag mode extra...   -> $R/<name>_<backend>_<tag>.*
  local name=$1 be=$2 tag=$3 mode=$4; shift 4
  local ngl=0; [[ $be == cuda ]] && ngl=99
  "$BIN/nsa-dump-$be" -m "$M" -t "$G/tokens_$name.txt" -o "$R/${name}_${be}_${tag}" --mode "$mode" -ngl $ngl "$@" 2> "$R/${name}_${be}_${tag}.log"
}

cmp() { # name tag
  local name=$1 tag=$2 extra=()
  if [[ -f $G/blockidx_$name.npy && -f $R/${name}_${tag}.idx.bin ]]; then
    extra=(--ref-idx "$G/blockidx_$name.npy" --ours-idx "$R/${name}_${tag}.idx.bin")
    [[ -f $G/margin_$name.npy ]] && extra+=(--margin "$G/margin_$name.npy")
  fi
  $CMP --ref "$G/logits_$name.npy" --ours "$R/${name}_${tag}.logits.bin" --label "$name $tag vs kernel-ref" \
       --json "$R/cmp_${name}_${tag}.json" "${extra[@]}" > /dev/null
  if [[ -f $G/logits_${name}_fp32.npy ]]; then
    # block sets too: the fp32 reference path has its own selections (no margins, so mismatches are only counted)
    local extra32=()
    if [[ -f $G/blockidx_${name}_fp32.npy && -f $R/${name}_${tag}.idx.bin ]]; then
      extra32=(--ref-idx "$G/blockidx_${name}_fp32.npy" --ours-idx "$R/${name}_${tag}.idx.bin")
    fi
    $CMP --ref "$G/logits_${name}_fp32.npy" --ours "$R/${name}_${tag}.logits.bin" --label "$name $tag vs fp32-ref" \
         --json "$R/cmp_${name}_${tag}_fp32.json" "${extra32[@]}" > /dev/null
  fi
}

# 1. op level: test-backend-ops (CUDA vs CPU) and the mask-based reference paths
"$CUDA_BUILD/bin/test-backend-ops" test -b CUDA0 -o NSA_SELECT   > "$R/test-backend-ops.txt" 2>&1
"$CUDA_BUILD/bin/test-backend-ops" test -b CUDA0 -o NSA_SEL_ATTN >> "$R/test-backend-ops.txt" 2>&1
for cfg in "512 4096" "1 4096" "4 4096" "1024 32768"; do "$BIN/nsa-mask-ref" $cfg; done > "$R/nsa-mask-ref.txt" 2>&1

# 2. model level: CPU and CUDA prefill (+ block indices, greedy continuation)
for name in short len600 len2k len4k; do
  [[ -f $G/tokens_$name.txt && -f $G/logits_$name.npy ]] || continue
  run $name cpu  prefill prefill --idx
  run $name cuda prefill prefill --idx --gen 32
  cmp $name cpu_prefill
  cmp $name cuda_prefill
done

# 3. prefill vs token-by-token decode (NSA decode path) and 512-token chunks; dense-mask SWA reference
run len2k cuda decode decode --idx
run len2k cuda chunk  chunk --chunk 512 --idx
LLAMA_NSA_DENSE_SWA=1 run len2k cuda denseswa prefill --idx
cmp len2k cuda_decode
cmp len2k cuda_chunk
cmp len2k cuda_denseswa

# 4. layer level: hidden states after every layer (len2k)
run len2k cpu  hidden prefill --hidden
run len2k cuda hidden prefill --hidden

# 5. GPU fallback check: backend of every graph node (GGML_SCHED_DEBUG=2), prefill and 1-token decode
GGML_SCHED_DEBUG=2 run len600 cuda sched_prefill prefill
GGML_SCHED_DEBUG=2 run len600 cuda sched_decode chunk --chunk 1
$PY -I "$ROOT/tests/gdn-nsa-test/analysis/sched_summary.py" "$R/len600_cuda_sched_prefill.log" "$R/len600_cuda_sched_decode.log" > "$R/sched_summary.txt"

# 6. quantized KV cache (q8_0, q4_0): the same runs with -ctk/-ctv, compared with the fp32 reference and, when
#    present, with the fake-quant reference (logits_NAME_fp32_kvTYPE.npy: K/V quantized + dequantized as ggml does)
kvrun() { # out be name mode ctk ctv extra...   -> $R/kvq/<out>.*
  local out=$1 be=$2 name=$3 mode=$4 ctk=$5 ctv=$6; shift 6
  local ngl=0; [[ $be == cuda ]] && ngl=99
  "$BIN/nsa-dump-$be" -m "$M" -t "$G/tokens_$name.txt" -o "$R/kvq/$out" --mode "$mode" -ngl $ngl -ctk $ctk -ctv $ctv --idx "$@" \
      2> "$R/kvq/$out.log"
}
mkdir -p "$R/kvq"
for name in len600 len2k len4k; do
  [[ -f $G/tokens_$name.txt ]] && kvrun ${name}_cuda_f16 cuda $name prefill f16 f16
done
kvrun len2k_cpu_f16 cpu len2k prefill f16 f16
for kv in q8_0 q4_0; do
  for name in len600 len2k len4k; do
    [[ -f $G/tokens_$name.txt ]] && kvrun ${name}_cuda_$kv cuda $name prefill $kv $kv
  done
  kvrun len2k_cuda_${kv}_decode cuda len2k decode  $kv $kv
  kvrun len2k_cuda_${kv}_chunk  cuda len2k chunk   $kv $kv --chunk 512
  kvrun len2k_cpu_$kv           cpu  len2k prefill $kv $kv
  LLAMA_ATTN_ROT_ENABLE=1 kvrun len2k_cuda_${kv}_rot        cuda len2k prefill $kv $kv
  LLAMA_ATTN_ROT_ENABLE=1 kvrun len2k_cuda_${kv}_rot_decode cuda len2k decode  $kv $kv
  LLAMA_NSA_DENSE_SWA=1   kvrun len2k_cuda_${kv}_denseswa   cuda len2k prefill $kv $kv
  GGML_SCHED_DEBUG=2 kvrun len600_cuda_${kv}_sched_prefill cuda len600 prefill $kv $kv
  GGML_SCHED_DEBUG=2 kvrun len600_cuda_${kv}_sched_decode  cuda len600 chunk   $kv $kv --chunk 1
  $PY -I "$ROOT/tests/gdn-nsa-test/analysis/sched_summary.py" "$R/kvq/len600_cuda_${kv}_sched_prefill.log" "$R/kvq/len600_cuda_${kv}_sched_decode.log" \
      > "$R/kvq/sched_summary_$kv.txt"
done
[[ -f $G/tokens_len4k.txt ]] && kvrun len4k_cuda_q8_0_q4_0 cuda len4k prefill q8_0 q4_0
$PY -I "$ROOT/tests/gdn-nsa-test/analysis/kvq_summary.py" "$R/kvq" "$G" > "$R/kvq/summary.md"

echo "results in $R"
