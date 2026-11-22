# gdn-nsa: correctness validation against the reference (L0-L5)

Overview, performance and commands: [README.md](../../README.md)

## Contents

- [Summary](#summary)
- [1. Reference](#1-reference)
- [2. L0: GGUF conversion](#2-l0-gguf-conversion)
- [3. L1: operators](#3-l1-operators)
- [4. L2: layers](#4-l2-layers)
- [5. L3: prefill](#5-l3-prefill)
- [6. L4: decode and cache state](#6-l4-decode-and-cache-state)
- [7. L5: end to end](#7-l5-end-to-end)
- [8. Functional benchmark: RULER S-NIAH subset](#8-functional-benchmark-ruler-s-niah-subset)
- [9. Gaps](#9-gaps)
- [10. Reproduce](#10-reproduce)

## Summary

The port is validated against the original FLA (PyTorch/Triton) model in six levels. Each level measures its own
error, so that a final token match cannot hide a wrong state, and a near-tie flip cannot look like a broken op.

| Level | What is compared | Result |
|---|---|---|
| L0 | GGUF tensors vs the HF checkpoint | 290/290 tensors bitwise identical; 205/205 layout tests; 30/30 metadata keys |
| L1 | GDN op vs a fp64 oracle; NSA ops vs fla on identical tensors | GDN: 636/636 pass, NRMSE ~1e-7, prefill and decode paths bitwise equal. NSA: 576/576 pass, F32 NRMSE ≤ 2e-6 outside the CUDA TF32 cuBLAS branches (3.3e-4 there) |
| L2 | Hidden state after each of the 24 layers (len2k) | NRMSE 1.1e-4 → 3.3e-4 (CPU), 1.5e-4 → 6.1e-4 (CUDA); smooth growth, no jump |
| L3 | Prefill logits at every position, incl. block/window boundary lengths | top-1 100% up to 2048 tokens, KL ≤ 6e-7 (6-token short prompt on CUDA: 2.4e-6); boundary lengths 1…513 all pass |
| L4 | Prefill(S) + token-by-token decode vs full prefill and reference | S = 1, 63, 64, 65, 127, 128, 511, 512, 513: top-1 100%, KL ≤ 3.3e-7 |
| L5 | CPU vs GPU, greedy generation, GPU fallback | CPU vs GPU KL ≤ 5.3e-7 (short prompt: 2.4e-6); greedy 32/32; only the embedding GET_ROWS runs on CPU |
| Functional | RULER S-NIAH subset (9 tasks × 20 samples), FLA vs ggml | FLA 65.0 = ggml CPU 65.0 = ggml GPU 65.0 (paper 74.8); per-sample outcome identical on 180/180 |

The automated report has 37 checks, all PASS (`tests/gdn-nsa-test/scripts/run_levels.sh` → `report.md`).

Pass criteria were set before the runs:

- **Logits** (fixed thresholds):
  - NRMSE ≤ 2e-3
  - mean KL ≤ 1e-5
  - top-1 ≥ 99.5%
  - every top-1 mismatch must be a reference near tie, with top1-top2 margin < 0.05
- **Block sets:** ≥ 99% must match exactly, and every mismatch must have a reference score margin < 1e-3.
- **Per-op thresholds:** see each section.

All measured values are within these limits, but the headroom differs by item. Some are two to three orders of magnitude below (GDN output NRMSE about 1e-7 against 1e-4). Others are only a few times below: logit NRMSE (at most 5.3e-4 against 2e-3), layer NRMSE (at most 6.1e-4 against 2e-3) and F16 K/V NRMSE (at most 6.2e-4 against 1e-3).

## 1. Reference

`reference_manifest.json`, written by `levels_report.py`, records the full reference environment:

| | |
|---|---|
| Model | Idiap/gated-deltanet-nsa-0.4B-10B, revision `c283b044b5688da94e46ebbd2c50078c2c9b11ef`, safetensors sha256 `409ac2b3…3c27986` |
| Reference code | idiap/hybrid-linear-sparse-attention @ `b786cbc6`, unmodified |
| Stack | torch 2.7.0+cu128, transformers 4.49.0, triton 3.3.0, flash-attn 2.8.3 |
| Precision | **kernel** reference: the author's path, i.e. fp32 weights, q/k/v cast to fp16, Triton NSA/GDN kernels, flash-attn window. **fp32** reference: pure PyTorch float32, `naive_nsa` + SDPA window + float32 recurrent GDN. TF32 disabled |
| Decoding | greedy, eval mode |
| Hardware | NVIDIA B200 (driver 595.58.03), Intel Xeon Platinum 8570 |

The kernel path is fp16 inside the layers, so two references are kept. The fp32 reference is the *mathematical* one,
the kernel reference is the *production* one. The port is compared with both. At L1 the oracle is either fp64
(GDN) or fla's naive fp32 code run in fp64/fp32 (NSA).

## 2. L0: GGUF conversion

**Tool:** `tests/gdn-nsa-test/analysis/check_gguf_weights.py`. It re-implements the converter mapping independently; it does not
import the converter, so a converter bug cannot be copied into the check.

- **Bitwise:** every GGUF tensor is rebuilt from the HF tensors and compared as uint32. This covers:
  - the fused q/k/v and conv tensors
  - `ssm_a = -exp(A_log)`, computed in float32
  - `dt_bias → ssm_dt.bias`

  Result: 290/290 equal, max abs 0.
- **Coverage:** every one of the 338 HF tensors is used exactly once, and all 290 GGUF tensors are accounted for. There is no `output.weight` because the embeddings are tied.
- **Layout (projection) test:** every 2-D weight is read from the file with ggml strides (ne[0] = input features), and y = W·x is compared with the float64 HF reference. Result: 205/205 with relative error 0. As a negative control, reading a square weight transposed gives a relative error of about 1, so the test does catch transposes.
- **Metadata:** 30 keys are checked against config.json, including:
  - head counts and dims
  - the NSA block size, block count and window
  - rope
  - rms eps
  - n_ff = 2816 (fla GatedMLP rule)
  - ssm params
  - recurrent_layers
  - vocab
- **Self-test** (not recorded in `l0_weights.json`; the tool has no self-test option): three seeded faults were all caught by both the bitwise and the projection check:
  - q and k swapped
  - a transposed weight
  - a 1-ulp change

## 3. L1: operators

### 3.1 GDN: `ggml_gated_delta_net` vs fp64 oracle

**Tool:** `tests/gdn-nsa-test/cpp/gdn-oracle-test.cpp`, run on the CPU and CUDA backends.

**Which op the model runs.** Prefill and decode use the same op, `ggml_gated_delta_net` (fused path, K = 1). The CUDA kernel is
`gated_delta_net_cuda<128>`; the MMA chunked kernel needs H = 16+ and is never used for this model's 8 heads.

**Oracle.** A host fp64 recurrence written from the op's contract:

- S = e^g·S
- u = β(v − Sᵀk)
- S += k·uᵀ
- o = Sᵀq / √D

This matches fla's recurrence; the scale is on the output, which equals fla's default `scale = K^-0.5`. The oracle is
checked against closed forms, and three deliberately broken oracles are rejected: a transposed state, a missing scale,
and the decay applied in the wrong order.

**Results.** Model shape (H = 8, D = 128) and a small shape (H = 2, D = 16), random non-zero initial state,
T ∈ {1, 2, 8, 63, 64, 65, 128, 129, 512}:

| | T=1 | 64 | 65 | 128 | 512 |
|---|---|---|---|---|---|
| CUDA output NRMSE | 1.0e-7 | 1.4e-7 | 1.4e-7 | 1.5e-7 | 1.5e-7 |
| CUDA final state max abs | 6.5e-8 | 9.5e-8 | 1.6e-7 | 1.1e-7 | 1.4e-7 |
| CPU output NRMSE | 8.5e-8 | 1.1e-7 | 1.2e-7 | 1.2e-7 | 1.2e-7 |

- **Prefill vs decode.** A single call over T tokens and T single-token calls that carry the state give **bitwise identical** outputs and final states on both backends.
- **Edge cases.** All pass:
  - zero key
  - zero value
  - β = 0 and β = 1
  - g = 0 (no forgetting)
  - g = −60 and g = −1000 (state wiped)
  - zero vs non-zero initial state
- **Error growth.** With realistic gates the error stays flat at about 1e-7. With g = 0 (no forgetting) the state error grows roughly like T^0.6 and reaches 2.4e-6 at T = 512.
- **Thresholds.** Output NRMSE ≤ 1e-4; state max abs ≤ 1e-5.

### 3.2 NSA: new ops vs fla on identical tensors

**Tools:**
- `tests/gdn-nsa-test/reference/make_nsa_fixtures.py` builds fixtures with fla's own naive functions: `mean_pooling`, `naive_nsa_topk` (fp64), `naive_nsa_cmp`, `naive_nsa_sel` and `naive_nsa`, plus an fp32 SDPA window. fla's window branch is fp16-only flash-attn, so it is replaced the same way as in the fp32 reference.
- `tests/gdn-nsa-test/cpp/nsa-fixture-test.cpp` runs the ggml graphs on CPU and CUDA, with K/V in F32 and F16.

**How the KV cache is set up.** As in the model: N is padded to 256 and the rows ≥ T are filled with garbage, so masking is exercised.

**Cases.** Random inputs for T ∈ {1, 2, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1000, 2048}, plus a *deterministic selection* case (T = 2048, block scores separated by construction, so the top-16 must match exactly).

**What is checked, per branch:**

| Check | Meaning |
|---|---|
| a. selection | `ggml_nsa_select` vs `naive_nsa_topk` as index sets. A mismatch is allowed only when fla's own relative margin (s16 − s17)/s16 is below 1e-4 (F32) or 1e-3 (F16) |
| b. selected attention, **reference indices fixed** | `ggml_nsa_sel_attn(ref_idx)` vs `naive_nsa_sel`. This isolates the attention kernel from selection |
| c. selected attention, own indices | end-to-end selection + attention |
| d. sliding window | `ggml_nsa_sel_attn_ext` with the model's window block list vs the fp32 window; `p ≥ 512` rows checked separately |
| e. compression | the model's block-mean + dense attention + zero block vs `naive_nsa_cmp` |
| f. gate fusion | σ-gated sum vs `naive_nsa`, plus one-hot gates (1,0,0), (0,1,0), (0,0,1) |

**Results.** 576/576 checks pass with no NaN/Inf, and cosine ≥ 0.9999998 everywhere. Values are worst over the cases, CUDA:

| | F32 K/V | F16 K/V |
|---|---|---|
| a. selection exact | 100% (deterministic case 100%) | 100%, except rand_T2048: 2 of 4096 rows, fla margin ≤ 9.3e-6 |
| b. sel_attn, fixed ref idx (max abs / NRMSE) | 5.2e-6 / 1.4e-6 | 1.3e-3 / 5.6e-4 |
| d. sliding window | 2.7e-6 / 7.2e-7 | 1.3e-3 / 5.5e-4 |
| e. compression | CPU 9.9e-7; CUDA 2.1e-4 / 3.3e-4 (TF32, see below) | 2.2e-4 / 3.4e-4 |
| f. gated output | CPU 7.2e-6 / 1.2e-6; CUDA 1.9e-4 / 1.1e-4 (TF32) | 2.3e-3 / 4.5e-4 |

**Thresholds.**
- **F32:** max abs 1e-4 and NRMSE 1e-3.
- **F16 K/V:** max abs 4e-3 and NRMSE 1e-3. F16 rounding is u = 2^-11 ≈ 4.9e-4, and the measured NRMSE is about 1·u.

**One finding: TF32 on the CUDA compression branch.** On CUDA with F32 K/V, o_cmp is off by up to 2.1e-4, which fails the F32 limit for T ≥ 511 (T = 128 and 129: 1.7e-5 and 1.2e-5, within it).
- **Cause:** o_cmp is computed with ggml_mul_mat, and ggml-cuda creates its cuBLAS handles with `CUBLAS_TF32_TENSOR_OP_MATH` (`ggml/src/ggml-cuda/common.cuh`). That setting comes from the upstream base, not from this port, and `ggml_prec_set_acc(F32)` does not turn it off.
- **Proof:** with `NVIDIA_TF32_OVERRIDE=0` the error drops to at most 5.3e-7 (NRMSE 3e-7) and the strict F32 run passes as well (`l1_nsa_notf32.json`).
- **How it is judged:** only the cuBLAS branches (e, f) on CUDA with F32 K/V use a labelled TF32 class, max abs 5e-4 and NRMSE 1e-3. The new NSA ops are unaffected.
- **Consequence:** this is also why CUDA prefill logits are about 2× noisier than CPU (L3).

The existing op tests complement this (`tests/gdn-nsa-test/scripts/run_validation.sh`):
- `test-backend-ops` (CUDA vs CPU): NSA_SELECT 8/8, NSA_SEL_ATTN 101/101. Its select check uses a double-precision near-tie rule.
- `nsa-mask-ref`: the new ops vs a reimplementation with existing ggml ops, up to 32K context.

## 4. L2: layers

Hidden state after each layer, len2k (2048 tokens), vs the kernel reference:

| layer | 0 GDN | 1 NSA | 5 NSA | 9 NSA | 13 NSA | 17 NSA | 21 NSA | 23 NSA |
|---|---|---|---|---|---|---|---|---|
| CPU NRMSE | 1.1e-4 | 1.3e-4 | 1.7e-4 | 2.3e-4 | 2.7e-4 | 3.0e-4 | 3.3e-4 | 2.9e-4 |
| CUDA NRMSE | 1.5e-4 | 2.5e-4 | 3.5e-4 | 4.2e-4 | 4.9e-4 | 6.1e-4 | 5.9e-4 | 5.3e-4 |

The error grows smoothly with depth: the largest layer-to-layer NRMSE ratio is 1.20 on CPU and 1.63 on CUDA, with no
outlier layer. The minimum per-token cosine is 0.99990 (CUDA, layer 17; CPU 0.999986). The full table is in `report.md`.

## 5. L3: prefill

Logits at **every** position, not only the last:

| prompt | backend | NRMSE | min cos | KL mean | KL max | top-1 | KL vs fp32 ref |
|---|---|---|---|---|---|---|---|
| len600 | CPU | 8.6e-5 | 1.000000 | 8.0e-8 | 6.5e-7 | 100% | 2.6e-8 |
| len600 | CUDA | 2.1e-4 | 1.000000 | 3.7e-7 | 4.5e-6 | 100% | 3.0e-7 |
| len2k | CPU | 1.0e-4 | 0.999999 | 9.1e-8 | 5.4e-6 | 100% | 2.9e-8 |
| len2k | CUDA | 2.3e-4 | 0.999995 | 4.4e-7 | 1.2e-4 | 100% | 3.6e-7 |
| len4k | CPU | 1.7e-4 | 0.999987 | 1.6e-7 | 4.4e-5 | 99.975% | 4.4e-8 |
| len4k | CUDA | 2.9e-4 | 0.999985 | 6.0e-7 | 4.8e-4 | 99.950% | 5.3e-7 |

- **len4k top-1 misses.** Every miss is a reference near tie: top1-top2 margin ≤ 0.001. The reference's own kernel vs fp32 runs agree only 99.975% at len4k.
- **Selected block sets vs the kernel reference:**
  - len600: 100%
  - len2k: 99.92% (CPU) and 99.90% (CUDA)
  - len4k: 99.77% (CPU) and 99.74% (CUDA)
  - Every mismatch has a reference score margin ≤ 9.6e-5.
- **Boundary lengths** (`tests/gdn-nsa-test/cpp/nsa-cache-equiv.cpp`, "lengths"): the first L tokens of len600 are prefilled as one ubatch, for L ∈ {1, 2, 63, 64, 65, 127, 128, 129, 511, 512, 513}.
  - These cross the 64-token block boundary and the 512-token window boundary.
  - Each run is compared with the full prefill (causality) and with the reference rows.
  - Worst vs reference: NRMSE 8.4e-5 and KL 7.9e-8 on CPU; 2.2e-4 and 3.7e-7 on CUDA. Top-1 is 100% in every run.

## 6. L4: decode and cache state

**Prefill(S) + token-by-token decode** (`nsa-cache-equiv`, "splits"): the first S tokens of len600 are prefilled in one ubatch, then the remaining tokens are decoded one by one to position 600.
- **Splits:** S ∈ {1, 63, 64, 65, 127, 128, 511, 512, 513}.
  - S = 63/64/65: the first decode steps complete a compression block.
  - S = 511/512/513: the decode steps cross the sliding-window boundary.
  - S = 1: pure decode.
- **What is compared:** the decoded rows vs the full prefill of the same backend, and vs the reference:

| backend | worst vs full prefill (NRMSE / KL) | worst vs reference (NRMSE / KL) | top-1 |
|---|---|---|---|
| CPU | 1.4e-5 / 2.1e-9 | 9.6e-5 / 8.9e-8 | 100% at every split |
| CUDA | 2.1e-4 / 3.3e-7 | 1.0e-4 / 9.5e-8 | 100% at every split |

- **CUDA decode is closer to the reference than CUDA prefill.** Decode rows have no TF32 GEMM.
- **What this exercises:**
  - the GDN state carry (bitwise-checked at L1 as well)
  - the KV cell = position mapping
  - the block means read from the cache
  - the window offset at absolute positions
- **Longer decode:** the full 2048-token decode and the 512-token chunked runs of len2k also pass: decode NRMSE 1.0e-4, KL 8.9e-8, top-1 100%.

## 7. L5: end to end

- **CPU vs GPU (direct):** KL 3.1e-7 (len600), 3.7e-7 (len2k) and 5.3e-7 (len4k). Top-1 is 100%, 100% and 99.975%. The three comparisons (reference vs CPU, reference vs GPU, CPU vs GPU) fit together. CPU is closer to the reference (len2k KL 9e-8, against 4.4e-7 for CUDA), and the CPU-to-GPU distance (3.7e-7) is about the same as the GPU-to-reference distance. So the GPU difference is the TF32 noise of section 3.2, not a separate error.
- **Greedy generation:** 32 tokens on len2k with the cache, identical to HF `generate()`; there is no first divergence.
- **No silent CPU fallback:** `GGML_SCHED_DEBUG=2` node assignment for a prefill and a decode graph shows that the only nodes on CPU are two `GET_ROWS` nodes per graph (`embd` and `node_1` in `sched_summary.txt`). All `NSA_SELECT` / `NSA_SEL_ATTN` nodes, the GDN op and every matmul run on CUDA0.

## 8. Functional benchmark: RULER S-NIAH subset

**Why.** Numerical agreement (L0–L5) shows that the port computes the same function. A retrieval benchmark checks the
same thing functionally: the converted model has to answer the *same questions the same way* as the original, at 1K, 2K
and 4K context. The paper evaluates laNSA 0.4B on RULER S-NIAH (Table 2), so a subset of it is run here.

**Setup** (`tests/gdn-nsa-test/bench/`):
- **Samples:** 20 fixed samples per task and length, 180 in total. They are built by `make_ruler_niah.py` with RULER's own strings, fetched from NVIDIA/RULER (`constants.py`, `niah.py`, `synthetic.yaml`):
  - S1: noise haystack, 7-digit number
  - S2: Paul Graham essays, number
  - S3: essays, UUID

  The samples are stored as token ids, so all three backends get identical inputs.
- **Backends:**
  - FLA reference: HF `generate()`, the author's kernel path
  - ggml CPU and ggml GPU: `nsa-generate` with the F32 GGUF and an F16 KV cache, one-ubatch prefill
- **Decoding:** greedy, 64 new tokens, stop at EOS. All three are identical in this.
- **Scoring:** RULER `string_match_all` after RULER's `postprocess_pred`.

**Results.** Accuracy in %. Δ = ggml − FLA. agree = identical generated token sequence (tok) / same per-sample correctness (corr).

| Task | Paper | FLA | ggml CPU | ggml GPU | Δ CPU | Δ GPU | agree CPU (tok / corr) | agree GPU (tok / corr) |
|---|---|---|---|---|---|---|---|---|
| S1 1K | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 20/20 / 20/20 | 20/20 / 20/20 |
| S1 2K | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 20/20 / 20/20 | 20/20 / 20/20 |
| S1 4K | 65.0 | 55.0 | 55.0 | 55.0 | +0.0 | +0.0 | 20/20 / 20/20 | 20/20 / 20/20 |
| S2 1K | 88.6 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 20/20 / 20/20 | 20/20 / 20/20 |
| S2 2K | 100.0 | 75.0 | 75.0 | 75.0 | +0.0 | +0.0 | 20/20 / 20/20 | 19/20 / 20/20 |
| S2 4K | 37.6 | 30.0 | 30.0 | 30.0 | +0.0 | +0.0 | 18/20 / 20/20 | 18/20 / 20/20 |
| S3 1K | 97.6 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 19/20 / 20/20 | 19/20 / 20/20 |
| S3 2K | 73.0 | 15.0 | 15.0 | 15.0 | +0.0 | +0.0 | 20/20 / 20/20 | 20/20 / 20/20 |
| S3 4K | 11.8 | 10.0 | 10.0 | 10.0 | +0.0 | +0.0 | 20/20 / 20/20 | 20/20 / 20/20 |
| **Average** | 74.8 | **65.0** | **65.0** | **65.0** | **+0.0** | **+0.0** | 177/180 / **180/180** | 176/180 / **180/180** |

- **ggml equals FLA in every cell.** The per-sample outcome is the same for all 180 samples on CPU and on GPU, so Δ = 0 and the correctness agreement is 100%.
- **Token-sequence differences.** The generated token sequences differ in 3 samples (CPU) and 4 (GPU). In every one of them the split happens *after* the answer, or both sides fail the sample:
  - S2_4096_07: step 5, all fail
  - S2_4096_10: step 9, after "5107245."
  - S3_1024_14: step 40, after the full UUID
  - S2_2048_02: GPU only, step 11, after "4823498."

  This is the near-tie argmax drift that L3 also shows (top-1 99.95–100%).
- **The gap to the paper (74.8 → 65.0) is shared by FLA and ggml,** so it does not come from the port. The largest gaps are S3-2K (15 vs 73), S2-2K (75 vs 100) and S1-4K (55 vs 65). The likely causes are in our sample generator rather than the model (a hypothesis, not tested):
  - essay source and concatenation
  - fixed needle depths i/19 instead of RULER's random 40 depths
  - a 64-token budget instead of 128
  - no lm-eval instances
- **This is a subset, not a reproduction of the paper numbers.** With N = 20, one sample is 5 pp.
- **Generation budget.** At 32 new tokens, S3 is cut off: a UUID is 29–37 tokens (mostly 33–35) with this tokenizer, which splits digits. The scores are then 53.3 / 53.3 / 53.3, again with Δ = 0 (`ruler_report_32.md`).
- **EVAPORATE** (paper Table 3) was not run.

```bash
S=<dir with nltk + wonderwords>   # pip --target; the reference venv itself is unchanged
R=<output dir>
PYTHONPATH=$S/pylib NLTK_DATA=$S/nltk_data $PY tests/gdn-nsa-test/bench/make_ruler_niah.py --ckpt $CKPT --out $R --cache $R/src   # -> $R/ruler_niah.{jsonl,txt}
tests/gdn-nsa-test/scripts/build.sh                     # also builds nsa-generate-{cpu,cuda} into build-gdn-nsa/
build-gdn-nsa/nsa-generate-cuda -m $MODEL -i $R/ruler_niah.txt -o $R/gen_ggml_cuda.txt -ngl 99
build-gdn-nsa/nsa-generate-cpu  -m $MODEL -i $R/ruler_niah.txt -o $R/gen_ggml_cpu.txt  -t 64
$PY tests/gdn-nsa-test/bench/run_fla.py --ckpt $CKPT --data $R/ruler_niah.jsonl --out $R/gen_fla.txt --max-new 64
$PY tests/gdn-nsa-test/bench/score_ruler.py --dir $R --ckpt $CKPT   # -> $R/ruler_report.md, ruler.json (--trunc 32: 32-token scores)
```

## 9. Gaps

- **Real top-16 choice:** a real top-16 choice needs more than 16 candidates. That first happens at position 1024: 16 complete blocks plus the incomplete current block, and from there on whenever p mod 64 ≠ 63. Of the fixtures, only the T = 2048 cases reach it. The model-level block-set comparison covers up to 4000 positions.
- **Fixture coverage:** every fixture starts at position 0, and none uses a quantized KV cache or a transposed V. Those are covered by `test-backend-ops` (vs CPU) and by the model-level runs.
- **fla production output:** fla's Triton `parallel_nsa` (fp16) output is stored in the fixtures but not compared. Its max abs difference from fla's own naive fp32 code is mostly 2e-3 to 4e-3, and 7e-2 on rand_T2048.
- **Long-context error growth:** with no forgetting (g = 0), the GDN state error would pass 1e-5 absolute only after roughly 5–9K tokens (extrapolated from T ≤ 512). That is fp32 accumulation, not a bug; for longer contexts, judge state NRMSE instead (1.5e-7 at T = 65, 3.3e-7 at T = 512).

## 10. Reproduce

```bash
# inputs: MODEL=<f32 gguf> GOLDEN=<golden dir from reference/run_ref.py> CKPT=<HF checkpoint> REFREPO=<fla repo>
#         PY=<reference venv python>; FIXTURES=<NSA fixture dir> is created on first use (351 MB)
# model-level dumps (L2/L3/L5 inputs, op tests); the full command list is in the repository README.md (section 9)
tests/gdn-nsa-test/scripts/run_validation.sh <gpu> results
# L0, L1, L3/L4 and the combined report
MODEL_RESULTS=results tests/gdn-nsa-test/scripts/run_levels.sh <gpu> results-levels
```

| Level | Tool | Output |
|---|---|---|
| L0 | `check_gguf_weights.py` | `l0_weights.json` |
| L1 | `gdn-oracle-test.cpp`, `reference/make_nsa_fixtures.py` + `nsa-fixture-test.cpp` (built by `scripts/build.sh`), `test-backend-ops`, `nsa-mask-ref` | `l1_gdn.json`, `l1_nsa.json`, `l1_nsa_notf32.json` |
| L2, L3, L5 | `run_validation.sh` (`nsa-dump`), `levels_report.py` | `report.md` |
| L3, L4 | `nsa-cache-equiv.cpp` | `l34_{cpu,cuda}_len600.json` |
| all | `levels_report.py` | `report.md`, `reference_manifest.json` |
| functional | `bench/make_ruler_niah.py`, `nsa-generate.cpp`, `bench/run_fla.py`, `bench/score_ruler.py` | `bench/ruler_report.md`, `bench/ruler.json` |
