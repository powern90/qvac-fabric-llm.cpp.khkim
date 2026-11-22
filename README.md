# Track A: Gated DeltaNet + NSA 0.4B in qvac-fabric-llm.cpp

This submission adds native ggml/GGUF inference for
[Idiap/gated-deltanet-nsa-0.4B-10B](https://huggingface.co/Idiap/gated-deltanet-nsa-0.4B-10B) on **CPU and CUDA**.
Native Sparse Attention (NSA) is built on **two new first-class ggml ops**, and the same graph serves prompt processing
and token-by-token generation. No Python runs at inference time; only the converter is Python.

## Requirements at a glance

| # | Requirement | Status | Evidence |
|---|---|---|---|
| A | NSA as new first-class ggml ops; prompt processing and token-by-token generation | Done | Two new ops with CPU and CUDA kernels ([§1](#1-what-was-built)); token-by-token decode matches the reference as closely as prefill (L4, [§4](#4-correctness)) |
| 1 | Convert and load through GGUF | Done | `conversion/gdn_nsa.py` ([§2](#2-build-convert-run)); an independent check rebuilds all 290 GGUF tensors from the HF checkpoint, bitwise equal (L0) |
| 2 | Real output on CPU | Done | Same greedy text as on GPU. Perplexity 14.6316 vs reference 14.6317 ([§3](#3-output-on-cpu-and-gpu)) |
| 3 | Real output on GPU (CUDA), no silent CPU fallback | Done | Every NSA, GDN and matmul node runs on CUDA0; only the token-embedding lookup stays on the CPU. Exception: contexts beyond about 200K tokens ([§3](#3-output-on-cpu-and-gpu)) |
| 4 | Correctness vs the reference, methodology, tolerances | Done | Six levels (L0–L5) against two references, thresholds fixed before the runs ([§4](#4-correctness)) |
| 5 | Automated tests, reproducible evidence | Done | `test-backend-ops` 8/8 + 101/101 cases; validation report with 37 automated checks, all PASS ([§5](#5-tests-and-evidence)); every step as copy-paste commands ([§9](#9-run-everything-copy-paste)) |
| 6 | CPU/GPU performance, memory, limitations | Done | CPU and CUDA throughput and memory ([§6](#6-performance-and-memory)), limitations ([§7](#7-known-limitations)) |
| Bonus | Quantization | Done | Q8_0/Q4_0 KV cache read directly by the NSA kernels and checked against the fp32 reference; standard weight presets need no new kernels ([§8](#8-bonus-quantization)) |

## 1. What was built

**Model.** 24 layers: even layers are Gated DeltaNet (GDN, a linear-attention recurrence), odd layers are NSA. Hidden
size 1024, 358M parameters. Each NSA layer adds up three attention branches, each scaled by a learned gate:

| NSA branch | What it attends to | How it is computed |
|---|---|---|
| compressed | the mean K/V of every complete 64-token block | existing ggml ops (block mean + small dense attention) |
| selected | the tokens of the 16 highest-scoring blocks | **new op** `ggml_nsa_select` picks the blocks, **new op** `ggml_nsa_sel_attn` attends to them |
| sliding window | the last 512 tokens | `ggml_nsa_sel_attn_ext` (the same op with a lower position bound) over the 9 blocks that overlap the window |

**New ggml ops** (`GGML_OP_NSA_SELECT`, `GGML_OP_NSA_SEL_ATTN`):

- **Kernels:** CPU in `ggml/src/ggml-cpu/ops.cpp`, CUDA in `ggml/src/ggml-cuda/nsa.cu`. On CUDA, one thread block per (token, KV head) loads each selected K/V block into shared memory once for all 16 query heads that share it. Decode splits the selected blocks across thread blocks (split-K) to keep the GPU busy with a single token.
- The attention op reads the KV cache directly on both backends: F16/F32, or Q8_0/Q4_0.
- Block selection uses the same float operations on both backends, so CPU and CUDA pick bit-identical blocks for the same input.

**Prompt processing and generation** use the same graph (generation is one token per step). Each token's K/V goes to
the cache cell equal to its position, which the block math relies on.

**What goes where**

| Area | Files |
|---|---|
| new ops | `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/{ops.cpp,ops.h,ggml-cpu.c}`, `ggml/src/ggml-cuda/{nsa.cu,nsa.cuh,ggml-cuda.cu}`, `ggml/src/ggml-nsa-exp.h`, `ggml/include/ggml-rpc.h` |
| model | `src/models/gdn-nsa.cpp` (new), `src/models/models.h`, `src/llama-{arch,model,context,kv-cache}.cpp`, `src/llama-arch.h`, `src/llama-hparams.h` |
| converter | `conversion/gdn_nsa.py` (new), `conversion/__init__.py`, `gguf-py/gguf/{constants,gguf_writer}.py` |
| op tests | `tests/test-backend-ops.cpp` (`test_nsa_select`, `test_nsa_sel_attn`) |
| validation suite | `tests/gdn-nsa-test/` ([README](tests/gdn-nsa-test/README.md), [REPORT](tests/gdn-nsa-test/REPORT.md)), un-ignored in `tests/.gitignore` |

## 2. Build, convert, run

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100  # 100 = B200
cmake --build build-cuda -j --target llama llama-completion llama-bench test-backend-ops

hf download Idiap/gated-deltanet-nsa-0.4B-10B --local-dir ckpt
python convert_hf_to_gguf.py ckpt --outtype f32 --outfile gdn-nsa-0.4B-f32.gguf

P="The capital of France is Paris. The capital of Germany is"
build-cuda/bin/llama-completion -m gdn-nsa-0.4B-f32.gguf -p "$P" -n 24 --temp 0 -no-cnv -dev none   # CPU only
build-cuda/bin/llama-completion -m gdn-nsa-0.4B-f32.gguf -p "$P" -n 24 --temp 0 -no-cnv -ngl 99     # CUDA
```

The complete list of commands, including every validation step, is in [§9](#9-run-everything-copy-paste).

## 3. Output on CPU and GPU

Greedy output of the CPU build and of the CUDA build (`-ngl 99`; step 3 of [§9](#9-run-everything-copy-paste)), identical on both:

> The capital of France is Paris. The capital of Germany is **Berlin. The capital of Italy is Rome. The capital of Japan is Tokyo. The capital of the United States is Washington**

**Against the reference** (4000-token prompt; the authors' production model, see [§4](#4-correctness)):

| | Reference | CPU | CUDA |
|---|---|---|---|
| Perplexity | 14.6317 | 14.6316 | 14.6317 |
| Same top-1 token as the reference | — | 99.975% | 99.950% |

**No silent CPU fallback.** `GGML_SCHED_DEBUG=2` prints the backend of every graph node. In the prefill and decode
graphs of a 600-token run, all `NSA_SELECT` / `NSA_SEL_ATTN` nodes, the GDN op and every matmul run on CUDA0; the only
CPU nodes are the two token-embedding lookups (`GET_ROWS`), which llama.cpp keeps on the host by design. The NSA ops
never read weights and their CUDA kernels accept Q8_0/Q4_0 caches, so quantized weights or KV caches do not move them
to the CPU; KV cache types without a CUDA kernel are rejected when the context is created. One limit: beyond about 200K
tokens of context on the B200 the selection op no longer fits in shared memory and silently runs on the CPU
([§7](#7-known-limitations)).

## 4. Correctness

**Two references**, both from the authors' code (`idiap/hybrid-linear-sparse-attention` @ b786cbc, the fork of fla
that has this model type; versions pinned):

- **kernel reference:** the production path (fp32 weights, but fp16 q/k/v inside the layers and Triton kernels).
- **fp32 reference:** the same model in pure PyTorch float32.

The port is compared with both, because the kernel path itself is not exact.

**Six levels.** Each level measures its own error, so a final token match cannot hide a wrong intermediate state:

| Level | What is compared | Result |
|---|---|---|
| L0 | GGUF tensors vs HF checkpoint (independent checker) | 290/290 bitwise identical; 205/205 layout tests (a transposed weight is caught); 30/30 metadata keys |
| L1 | Each op alone: GDN vs an fp64 oracle; NSA ops vs fla's own code on identical tensors | GDN 636/636 checks, NRMSE ~1e-7; NSA 576/576 checks |
| L2 | Hidden state after each of the 24 layers (2K prompt) | NRMSE grows smoothly to at most 3.3e-4 (CPU) and 6.1e-4 (CUDA); no jump at any layer |
| L3 | Prefill logits at every position, incl. lengths at block/window boundaries | top-1 100% up to 2048 tokens, mean KL ≤ 6e-7 (2.4e-6 on a 6-token prompt) |
| L4 | Prefill of S tokens, then token-by-token decode (S = 1 … 513) | top-1 100%, KL ≤ 3.3e-7 |
| L5 | CPU vs GPU, greedy generation, GPU placement | CPU vs GPU KL ≤ 5.3e-7 (2.4e-6 on a 6-token prompt); 32/32 greedy tokens identical to HF `generate()` (CUDA, short and 2K prompts) |
| Functional | RULER single-needle retrieval (S-NIAH) subset, 180 samples at 1K/2K/4K | reference 65.0 = CPU 65.0 = GPU 65.0; same outcome on all 180 samples |

NRMSE = RMS error / RMS of the reference; KL = mean KL divergence of the next-token distributions; top-1 = how often
the most likely next token is the same.

**Tolerances**, fixed before the runs:

- **Logits:** NRMSE ≤ 2e-3, mean KL ≤ 1e-5, top-1 ≥ 99.5%. Every top-1 miss must be a near tie in the reference (top-1/top-2 logit margin < 0.05).
- **Selected blocks:** ≥ 99% of the block sets must match exactly, and every mismatch must be a near tie (score margin < 1e-3).
- **Single ops:** GDN output NRMSE ≤ 1e-4. NSA ops with F32 K/V: max abs ≤ 1e-4 and NRMSE ≤ 1e-3; with F16 K/V: max abs ≤ 4e-3, since F16 rounding alone is about 5e-4. The two CUDA branches that use cuBLAS get max abs ≤ 5e-4 with F32 K/V, because of TF32 (below).

**Why block selection needs a near-tie rule.** Picking the top 16 blocks is discrete: when two scores differ only by
float noise, either pick is valid. At 4K the port matches the kernel reference (fp16 q/k/v) on 99.77% (CPU) and 99.74%
(CUDA) of block sets, and the fp32 reference more often (99.97% CPU, 99.89% CUDA). Every mismatch with the kernel
reference has a score margin below 1e-4.

**CUDA prefill is about 2× further from the reference than CPU** (logit NRMSE 2.3e-4 vs 1.0e-4, 2K prompt). The cause
is TF32, a lower-precision tensor-core mode that the existing CUDA backend uses for F32 cuBLAS matmuls, not the new
kernels: without TF32 the op errors drop to the F32 level, and CUDA decode (no TF32 matmul) is as close as CPU.

Full method and every number: [tests/gdn-nsa-test/REPORT.md](tests/gdn-nsa-test/REPORT.md).

## 5. Tests and evidence

- **Op tests** (`tests/test-backend-ops.cpp`, CUDA vs CPU):
  - `NSA_SELECT`: 8/8, including decode, positions with no complete block, and large query groups; also checked against block scores recomputed in double precision.
  - `NSA_SEL_ATTN`: 101/101, including F16/F32/Q8_0/Q4_0 K/V, decode with split-K, the sliding window, odd shapes, and the transposed V cache layout.
  ```bash
  build-cuda/bin/test-backend-ops test -b CUDA0 -o NSA_SELECT
  build-cuda/bin/test-backend-ops test -b CUDA0 -o NSA_SEL_ATTN
  ```
- **Independent op check** (`nsa-mask-ref`): recomputes the new ops with existing ggml ops only (dense masked attention), up to a 32K context, on both backends. Attention agrees to a normalized squared error ≤ 2.1e-7; CPU and CUDA selections are bit-identical.
- **Validation suite** (`tests/gdn-nsa-test/`, [README](tests/gdn-nsa-test/README.md)):
  - `scripts/run_validation.sh`: model-level runs against the stored reference outputs, about 9 minutes on one GPU.
  - `scripts/run_levels.sh`: runs the L0–L4 checks and writes `report.md`, 37 PASS/FAIL checks over L0–L5.
  - The results of the final run are in `tests/gdn-nsa-test/results/`.
  - Generating the reference outputs needs the checkpoint and a separate PyTorch environment; the commands are in [§9](#9-run-everything-copy-paste).

## 6. Performance and memory

NVIDIA B200; F32 weights, F16 KV cache, `llama-bench -r 5` (`tests/gdn-nsa-test/results/model/bench_cuda.md`).

| CUDA (tokens/s) | Context 0 | Context 1024 | Context 3072 |
|---|---|---|---|
| Prompt processing, 2048 tokens | 39,919 | 35,895 | 37,139 |
| Generation, 128 tokens | 460 | 431 | 402 |

CPU: Intel Xeon Platinum 8570, 64 threads, same settings (`tests/gdn-nsa-test/results/model/bench_cpu.md`).

| CPU (tokens/s) | Context 0 | Context 3072 |
|---|---|---|
| Prompt processing, 2048 tokens | 1,099 | 1,039 |
| Generation, 64 tokens | 58 | 22 |

- **On CUDA, generation slows only 13% from 0 to 3072 tokens of context** (460 → 402 tokens/s). At a 4K context an NSA query reads about 1.7K K/V rows (selected blocks, window, block means) instead of 4K.
- **Routing the sliding window through the sparse op** instead of a dense masked attention makes prefill 1.38× faster at 3072 tokens of context (37.1K vs 26.9K tokens/s).

**Memory** (4096-token context, one sequence):

| | Size |
|---|---|
| Weights (F32 GGUF) | 1367 MiB |
| KV cache, F16 (12 NSA layers) | 24 MiB (6 KiB per token) |
| GDN recurrent state (12 GDN layers) | 6.42 MiB |

## 7. Known limitations

- **One sequence per context.** The block math needs KV cell = position. `n_seq_max > 1` is rejected with an error; context shift is not supported.
- **Kernels exist for CPU and CUDA only.** On any other GPU backend the two NSA ops would run on the CPU.
- **KV cache types:** F16/F32, or any pair of F16/Q8_0/Q4_0. Other types are rejected with an error that names the supported ones.
- **Block means are recomputed** over the whole cache at every step. This is cheap (O(context) per layer) but redundant.
- **Long contexts:** the CUDA selection op keeps all block scores in shared memory, which on the B200 fits about 200K tokens (fewer on GPUs with less shared memory). Beyond that it runs on the CPU without a warning. The model was trained at 4K.
- **Near ties:** at the model level, CPU and CUDA can still pick different blocks where two scores are almost equal, because their inputs differ slightly (TF32 on CUDA).
- **Tokenizer:** the checkpoint ships only `tokenizer.json`, mapped to llama.cpp's SentencePiece tokenizer with merge ranks as scores. This approximates BPE. The validation runs feed token ids from the HF tokenizer, so they do not test this mapping.
- **Quantized weights** are not part of the shipped results: `results/` covers F32 weights with F16, Q8_0 and Q4_0 KV caches.

## 8. Bonus: quantization

**Weights** (standard `llama-quantize` presets such as Q8_0, Q6_K or Q4_K_M; the NSA ops never read weights, so no new
kernels are needed). Step 10 of [§9](#9-run-everything-copy-paste) quantizes the model and compares it with the fp32
reference on the 4K prompt; those runs are not part of `results/`.

**KV cache** (`-ctk` / `-ctv`; the NSA kernels read the quantized cache directly). CUDA, 4000-token prompt, against the
fp32 reference (`tests/gdn-nsa-test/results/model/kvq_summary.md`):

| KV cache | Size at 4096 tokens | Δ perplexity | Same top-1 token as the fp32 reference |
|---|---|---|---|
| F16 | 24.00 MiB | +0.00% | 99.98% |
| Q8_0 | 12.75 MiB | +0.00% | 99.70% |
| Q4_0 | 6.75 MiB | +0.51% | 95.95% |
| K Q8_0, V Q4_0 | 9.75 MiB | +0.12% | 98.32% |

- Attention over a quantized cache is bit-identical to attention over the same values dequantized to F32, on CPU and CUDA.
- Against a PyTorch reference that quantizes K/V exactly like ggml, the remaining difference is about the same as between the port's own CPU and CUDA runs (Q8_0, 2K prompt: KL 5.4e-6 vs 5.6e-6).

## 9. Run everything (copy-paste)

Run the blocks in order from the repository root, in one shell. Only the variables in step 0 need editing. Each
step ends with the command that shows its result.

```bash
# 0. settings: work dir (checkpoint, reference env, outputs), GPU index (nvidia-smi), CPU threads
export WORK=$HOME/gdn-nsa-work GPU=0 THREADS=64 CUDA_ARCH=100   # CUDA_ARCH 100 = B200
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader | tr -d ' ')
mkdir -p $WORK
```

```bash
# 1. build: CPU, CUDA (arch 100 = B200; set yours) and the validation tools (-> build-gdn-nsa/)
cmake -S . -B build-cpu  -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
cmake --build build-cpu  -j --target llama llama-completion llama-bench test-backend-ops
cmake --build build-cuda -j --target llama llama-completion llama-bench llama-quantize test-backend-ops
tests/gdn-nsa-test/scripts/build.sh
```

```bash
# 2. python 3.11 environment (also the reference environment of step 5), download and convert to GGUF
conda create -y -p $WORK/env python=3.11        # or: python3.11 -m venv $WORK/env (the flash-attn wheel is cp311)
$WORK/env/bin/pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
$WORK/env/bin/pip install transformers==4.49.0 datasets==5.0.1 einops==0.8.2 safetensors==0.8.0 numpy==2.4.6 sentencepiece
$WORK/env/bin/hf download Idiap/gated-deltanet-nsa-0.4B-10B --revision c283b044b5688da94e46ebbd2c50078c2c9b11ef --local-dir $WORK/ckpt
$WORK/env/bin/python convert_hf_to_gguf.py $WORK/ckpt --outtype f32 --outfile $WORK/gdn-nsa-0.4B-f32.gguf
export MODEL=$WORK/gdn-nsa-0.4B-f32.gguf
```

```bash
# 3. real output: CPU build, then CUDA build (the same text is expected)
P="The capital of France is Paris. The capital of Germany is"
build-cpu/bin/llama-completion  -m $MODEL -p "$P" -n 24 --temp 0 -no-cnv -t $THREADS 2>/dev/null
build-cuda/bin/llama-completion -m $MODEL -p "$P" -n 24 --temp 0 -no-cnv -ngl 99 2>/dev/null
```

```bash
# 4. op tests, CUDA vs CPU (expect "8/8 tests passed" and "101/101 tests passed")
build-cuda/bin/test-backend-ops test -b CUDA0 -o NSA_SELECT
build-cuda/bin/test-backend-ops test -b CUDA0 -o NSA_SEL_ATTN
```

```bash
# 5. reference code (flash-attn, the authors' fla fork) and the reference outputs ("golden")
$WORK/env/bin/pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
git clone https://github.com/idiap/hybrid-linear-sparse-attention $WORK/ref-repo
git -C $WORK/ref-repo checkout b786cbc625c0427a5becaa4ff11d9f73ecbf42eb
$WORK/env/bin/pip install --no-deps -e $WORK/ref-repo
export PY=$WORK/env/bin/python GOLDEN=$WORK/golden GDN_NSA_CKPT=$WORK/ckpt GDN_NSA_GOLDEN=$WORK/golden
mkdir -p $GOLDEN
$PY tests/gdn-nsa-test/reference/check_flash_attn.py          # flash-attn window vs fp32 SDPA
$PY tests/gdn-nsa-test/reference/prep_prompts.py              # prompts: tokens_{short,len600,len2k,len4k}.txt
$PY tests/gdn-nsa-test/reference/run_ref.py --stage kernel --prompts short,len600,len2k
$PY tests/gdn-nsa-test/reference/run_ref.py --stage kernel --prompts len4k
$PY tests/gdn-nsa-test/reference/run_ref.py --stage fp32   --prompts short,len600
$PY tests/gdn-nsa-test/reference/run_ref.py --stage fp32   --prompts len2k,len4k
$PY tests/gdn-nsa-test/reference/run_ref.py --stage greedy --prompts short
$PY tests/gdn-nsa-test/reference/run_ref.py --stage greedy --prompts len2k
for kv in q8_0 q4_0; do $PY tests/gdn-nsa-test/reference/run_ref.py --stage fp32 --kv-fake-quant $kv --prompts short,len600,len2k,len4k; done
$PY tests/gdn-nsa-test/reference/run_ref.py --stage compare   # reference kernel vs reference fp32
ls $GOLDEN
```

```bash
# 6. model level (L2, L3, L5, quantized KV cache) against the reference, about 9 minutes
tests/gdn-nsa-test/scripts/run_validation.sh $GPU $WORK/results
$PY tests/gdn-nsa-test/analysis/summarize.py $WORK/results $GOLDEN   # logits, block sets, greedy, decode, perplexity
cat $WORK/results/nsa-mask-ref.txt $WORK/results/sched_summary.txt  # existing-op check; nodes per backend
cat $WORK/results/kvq/summary.md                                    # quantized KV cache
```

```bash
# 7. level checks L0-L4 and the PASS/FAIL report (37 checks); the NSA fixtures (351 MB) are made on first use
CKPT=$WORK/ckpt REFREPO=$WORK/ref-repo MODEL_RESULTS=$WORK/results FIXTURES=$WORK/fixtures/nsa \
  tests/gdn-nsa-test/scripts/run_levels.sh $GPU $WORK/results-levels
cat $WORK/results-levels/report.md
```

```bash
# 8. functional benchmark: RULER S-NIAH subset (180 samples; the CPU part is the slowest step)
$WORK/env/bin/pip install --target $WORK/ruler/pylib nltk==3.10.4 wonderwords==3.0.1
PYTHONPATH=$WORK/ruler/pylib $PY -m nltk.downloader -d $WORK/ruler/nltk_data punkt punkt_tab
R=$WORK/ruler
PYTHONPATH=$R/pylib NLTK_DATA=$R/nltk_data $PY tests/gdn-nsa-test/bench/make_ruler_niah.py --ckpt $WORK/ckpt --out $R --cache $R/src
build-gdn-nsa/nsa-generate-cuda -m $MODEL -i $R/ruler_niah.txt -o $R/gen_ggml_cuda.txt -ngl 99
build-gdn-nsa/nsa-generate-cpu  -m $MODEL -i $R/ruler_niah.txt -o $R/gen_ggml_cpu.txt  -t $THREADS
$PY tests/gdn-nsa-test/bench/run_fla.py --ckpt $WORK/ckpt --data $R/ruler_niah.jsonl --out $R/gen_fla.txt --max-new 64
$PY tests/gdn-nsa-test/bench/score_ruler.py --dir $R --ckpt $WORK/ckpt
cat $R/ruler_report.md
```

```bash
# 9. performance: CUDA, CPU, quantized KV cache, single ops
build-cuda/bin/llama-bench -m $MODEL -ngl 99 -p 512,2048 -n 128 -d 0,1024,3072 -r 5
build-cpu/bin/llama-bench  -m $MODEL -t $THREADS -p 512,2048 -n 64 -d 0,3072 -r 5
for kv in f16 q8_0 q4_0; do build-cuda/bin/llama-bench -m $MODEL -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 2048 -n 128 -d 0,3072,16384,32768 -r 5; done
LLAMA_NSA_DENSE_SWA=1 build-cuda/bin/llama-bench -m $MODEL -ngl 99 -p 2048 -n 128 -d 3072 -r 5   # dense-window comparison
build-cuda/bin/test-backend-ops perf -b CUDA0 -o NSA_SELECT
build-cuda/bin/test-backend-ops perf -b CUDA0 -o NSA_SEL_ATTN
```

```bash
# 10. quantized weights, e.g. Q4_K_M: quantize, run the 4K prompt, compare with the fp32 reference
build-cuda/bin/llama-quantize $MODEL $WORK/gdn-nsa-0.4B-Q4_K_M.gguf Q4_K_M
build-gdn-nsa/nsa-dump-cuda -m $WORK/gdn-nsa-0.4B-Q4_K_M.gguf -t $GOLDEN/tokens_len4k.txt -o $WORK/q4_k_m --mode prefill -ngl 99
$PY tests/gdn-nsa-test/analysis/compare.py --ref $GOLDEN/logits_len4k_fp32.npy --ours $WORK/q4_k_m.logits.bin
```

The results of the author's run of steps 4, 6, 7 and 8 are in `tests/gdn-nsa-test/results/` (`model/`, `levels/`,
`bench/`; step 6's `kvq/summary.md` is `model/kvq_summary.md`), so they can be compared with yours.
