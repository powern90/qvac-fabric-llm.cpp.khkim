#!/usr/bin/env bash
# Builds every C++ validation tool against the existing CPU and CUDA builds of this tree (the libs are not rebuilt).
#   [CPU_BUILD=dir] [CUDA_BUILD=dir] tests/gdn-nsa-test/scripts/build.sh [out_dir]      (default out: build-gdn-nsa/)
#   llama API tools, one binary per backend (-cpu / -cuda): nsa-dump, nsa-cache-equiv, nsa-generate
#   ggml-only tools, one binary that drives the CPU and CUDA backends: nsa-mask-ref, gdn-oracle-test, nsa-fixture-test
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$ROOT/tests/gdn-nsa-test/cpp
OUT=${1:-$ROOT/build-gdn-nsa}
mkdir -p "$OUT"
CPU_BUILD=${CPU_BUILD:-$ROOT/build-cpu}
CUDA_BUILD=${CUDA_BUILD:-$ROOT/build-cuda}

for b in cpu cuda; do
  if [[ $b == cpu ]]; then B=$CPU_BUILD/bin; else B=$CUDA_BUILD/bin; fi
  [[ -f $B/libllama.so ]] || { echo "skip $b: $B/libllama.so not found"; continue; }
  for t in nsa-dump nsa-cache-equiv nsa-generate; do
    g++ -O2 -std=c++17 -I"$ROOT/include" -I"$ROOT/ggml/include" "$SRC/$t.cpp" \
        -L"$B" -lllama -lggml -lggml-base -Wl,-rpath,"$B" -o "$OUT/$t-$b"
  done
done

B=$CUDA_BUILD/bin; [[ -f $B/libggml-cuda.so ]] || B=$CPU_BUILD/bin
for t in nsa-mask-ref gdn-oracle-test nsa-fixture-test; do
  g++ -O2 -std=c++17 -I"$ROOT/ggml/include" "$SRC/$t.cpp" \
      -L"$B" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$B" -o "$OUT/$t"
done
echo "tools in $OUT"
