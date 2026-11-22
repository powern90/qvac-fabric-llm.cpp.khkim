# gdn-nsa correctness report (L0-L5)

Pass criteria (declared up front): layer_nrmse 0.002, logit_nrmse 0.002, kl_mean 1e-05, top1 0.995, tie_margin 0.05, block_exact 0.99, block_tie 0.001

| level | check | backend | result | detail |
|---|---|---|---|---|
| L0 | GGUF tensors vs checkpoint (bitwise, coverage, layout/projection, metadata) | - | PASS | {"tensors_bitwise": {"pass": 290, "fail": 0}, "coverage": {"pass": 6, "fail": 0}, "projection": {"pass": 205, "fail": 0}, "metadata": {"pass": 30, "fail": 0}} |
| L1 | NSA ops vs fla on identical fixtures (selection, fixed-index sel_attn, SWA, cmp, gates) | cpu+cuda | PASS |  |
| L1 | GDN op vs fp64 oracle, prefill vs token-by-token state carry, edge cases | cpu+cuda | PASS | 636 checks, 0 failed; op ggml_gated_delta_net(q,k,v,g,beta,state,K=1); thresholds {"out_nrmse": 0.0001, "state_max_abs": 1e-05, "zero_ref_max_abs": 1e-06} |
| L1 | test-backend-ops NSA_SELECT / NSA_SEL_ATTN (CUDA vs CPU) | cuda | PASS | 8/8 tests passed; 101/101 tests passed |
| L1 | nsa-mask-ref (new ops vs existing-op reference, up to 32K) | cpu+cuda | PASS | 28 OK lines |
| L2 | hidden state after each of the 24 layers (len2k) | cpu | PASS | NRMSE max 3.30e-04, min cosine 0.999986, largest layer-to-layer NRMSE ratio 1.20 |
| L2 | hidden state after each of the 24 layers (len2k) | cuda | PASS | NRMSE max 6.08e-04, min cosine 0.999900, largest layer-to-layer NRMSE ratio 1.63 |
| L3 | prefill logits, all 6 positions (short) vs kernel ref | cpu | PASS | NRMSE 4.87e-05, KL 1.7e-08, top-1 100.000%, max tie margin 0.000 |
| L3 | selected block sets (short) vs kernel ref | cpu | PASS | exact 100.000% of 144, mismatches 0, max ref margin of a mismatch 0.0e+00 |
| L3 | prefill logits, all 6 positions (short) vs kernel ref | cuda | PASS | NRMSE 5.33e-04, KL 2.4e-06, top-1 100.000%, max tie margin 0.000 |
| L3 | selected block sets (short) vs kernel ref | cuda | PASS | exact 100.000% of 144, mismatches 0, max ref margin of a mismatch 0.0e+00 |
| L5 | CPU vs GPU logits (short) | cpu-cuda | PASS | NRMSE 5.32e-04, KL 2.4e-06, top-1 100.000% |
| L3 | prefill logits, all 600 positions (len600) vs kernel ref | cpu | PASS | NRMSE 8.59e-05, KL 8.0e-08, top-1 100.000%, max tie margin 0.000 |
| L3 | selected block sets (len600) vs kernel ref | cpu | PASS | exact 100.000% of 14400, mismatches 0, max ref margin of a mismatch 0.0e+00 |
| L3 | prefill logits, all 600 positions (len600) vs kernel ref | cuda | PASS | NRMSE 2.10e-04, KL 3.7e-07, top-1 100.000%, max tie margin 0.000 |
| L3 | selected block sets (len600) vs kernel ref | cuda | PASS | exact 100.000% of 14400, mismatches 0, max ref margin of a mismatch 0.0e+00 |
| L5 | CPU vs GPU logits (len600) | cpu-cuda | PASS | NRMSE 1.96e-04, KL 3.1e-07, top-1 100.000% |
| L3 | prefill logits, all 2048 positions (len2k) vs kernel ref | cpu | PASS | NRMSE 1.03e-04, KL 9.1e-08, top-1 100.000%, max tie margin 0.000 |
| L3 | selected block sets (len2k) vs kernel ref | cpu | PASS | exact 99.919% of 49152, mismatches 40, max ref margin of a mismatch 3.3e-05 |
| L3 | prefill logits, all 2048 positions (len2k) vs kernel ref | cuda | PASS | NRMSE 2.31e-04, KL 4.4e-07, top-1 100.000%, max tie margin 0.000 |
| L3 | selected block sets (len2k) vs kernel ref | cuda | PASS | exact 99.896% of 49152, mismatches 51, max ref margin of a mismatch 3.3e-05 |
| L5 | CPU vs GPU logits (len2k) | cpu-cuda | PASS | NRMSE 2.12e-04, KL 3.7e-07, top-1 100.000% |
| L3 | prefill logits, all 4000 positions (len4k) vs kernel ref | cpu | PASS | NRMSE 1.68e-04, KL 1.6e-07, top-1 99.975%, max tie margin 0.000 |
| L3 | selected block sets (len4k) vs kernel ref | cpu | PASS | exact 99.772% of 96000, mismatches 219, max ref margin of a mismatch 8.7e-05 |
| L3 | prefill logits, all 4000 positions (len4k) vs kernel ref | cuda | PASS | NRMSE 2.89e-04, KL 6.0e-07, top-1 99.950%, max tie margin 0.001 |
| L3 | selected block sets (len4k) vs kernel ref | cuda | PASS | exact 99.744% of 96000, mismatches 246, max ref margin of a mismatch 9.6e-05 |
| L5 | CPU vs GPU logits (len4k) | cpu-cuda | PASS | NRMSE 2.55e-04, KL 5.3e-07, top-1 99.975% |
| L4 | len2k decode (2048 positions) vs kernel ref | cuda | PASS | NRMSE 1.03e-04, KL 8.9e-08, top-1 100.000% |
| L4 | len2k chunk (2048 positions) vs kernel ref | cuda | PASS | NRMSE 2.31e-04, KL 4.5e-07, top-1 100.000% |
| L3 | len2k denseswa (2048 positions) vs kernel ref | cuda | PASS | NRMSE 2.22e-04, KL 3.9e-07, top-1 100.000% |
| L3 | prefill of the first L tokens, L = 1,2,63,64,65,127,128,129,511,512,513 | cpu | PASS | worst vs full prefill: NRMSE 3.2e-06, KL 5.6e-11; worst vs ref: NRMSE 8.4e-05, KL 7.9e-08, min top-1 100.00% |
| L4 | prefill S tokens + token-by-token decode to 600, S = 1,63,64,65,127,128,511,512,513 | cpu | PASS | worst vs full prefill: NRMSE 1.4e-05, KL 2.1e-09; worst vs ref: NRMSE 9.6e-05, KL 8.9e-08, min top-1 100.00% |
| L3 | prefill of the first L tokens, L = 1,2,63,64,65,127,128,129,511,512,513 | cuda | PASS | worst vs full prefill: NRMSE 1.8e-04, KL 2.3e-07; worst vs ref: NRMSE 2.2e-04, KL 3.7e-07, min top-1 100.00% |
| L4 | prefill S tokens + token-by-token decode to 600, S = 1,63,64,65,127,128,511,512,513 | cuda | PASS | worst vs full prefill: NRMSE 2.1e-04, KL 3.3e-07; worst vs ref: NRMSE 1.0e-04, KL 9.5e-08, min top-1 100.00% |
| L5 | greedy generation (short), HF generate() with cache | cuda | PASS | 32 tokens compared, first divergence none |
| L5 | greedy generation (len2k), HF generate() with cache | cuda | PASS | 32 tokens compared, first divergence none |
| L5 | no silent CPU fallback (GGML_SCHED_DEBUG=2 node assignment, prefill + decode) | cuda | PASS | ops on CPU: ['GET_ROWS']; NSA_SELECT / NSA_SEL_ATTN all on CUDA0 |

## L2 per-layer hidden state error (len2k)

| layer | type | cpu NRMSE | cpu min cos | cuda NRMSE | cuda min cos |
|---|---|---|---|---|---|
| 0 | GDN | 1.10e-04 | 1.000000 | 1.53e-04 | 1.000000 |
| 1 | NSA | 1.26e-04 | 1.000000 | 2.49e-04 | 1.000000 |
| 2 | GDN | 1.51e-04 | 1.000000 | 2.81e-04 | 1.000000 |
| 3 | NSA | 1.53e-04 | 1.000000 | 3.06e-04 | 1.000000 |
| 4 | GDN | 1.70e-04 | 1.000000 | 3.34e-04 | 1.000000 |
| 5 | NSA | 1.68e-04 | 1.000000 | 3.46e-04 | 1.000000 |
| 6 | GDN | 1.89e-04 | 1.000000 | 3.74e-04 | 1.000000 |
| 7 | NSA | 1.85e-04 | 1.000000 | 3.79e-04 | 1.000000 |
| 8 | GDN | 1.96e-04 | 1.000000 | 3.91e-04 | 1.000000 |
| 9 | NSA | 2.27e-04 | 0.999995 | 4.23e-04 | 0.999995 |
| 10 | GDN | 2.34e-04 | 0.999996 | 4.34e-04 | 0.999996 |
| 11 | NSA | 2.54e-04 | 0.999997 | 4.63e-04 | 0.999997 |
| 12 | GDN | 2.59e-04 | 0.999997 | 4.70e-04 | 0.999997 |
| 13 | NSA | 2.65e-04 | 0.999997 | 4.94e-04 | 0.999996 |
| 14 | GDN | 2.61e-04 | 0.999998 | 4.83e-04 | 0.999997 |
| 15 | NSA | 2.82e-04 | 0.999998 | 5.08e-04 | 0.999997 |
| 16 | GDN | 2.67e-04 | 0.999999 | 4.83e-04 | 0.999998 |
| 17 | NSA | 2.99e-04 | 0.999996 | 6.08e-04 | 0.999900 |
| 18 | GDN | 2.85e-04 | 0.999998 | 5.78e-04 | 0.999924 |
| 19 | NSA | 2.87e-04 | 0.999997 | 5.68e-04 | 0.999935 |
| 20 | GDN | 2.85e-04 | 0.999998 | 5.55e-04 | 0.999941 |
| 21 | NSA | 3.30e-04 | 0.999986 | 5.85e-04 | 0.999949 |
| 22 | GDN | 3.15e-04 | 0.999988 | 5.67e-04 | 0.999954 |
| 23 | NSA | 2.90e-04 | 0.999991 | 5.30e-04 | 0.999959 |

## Logits vs reference (extended metrics)

| prompt | backend | path | rows | max abs | NRMSE | min cos | KL mean | KL max | top-1 | non-finite | KL vs fp32 ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| short | cpu | prefill | 6 | 2.67e-03 | 4.87e-05 | 1.000000 | 1.7e-08 | 2.9e-08 | 100.000% | 0 | 1.1e-08 |
| short | cuda | prefill | 6 | 1.45e-02 | 5.33e-04 | 1.000000 | 2.4e-06 | 7.7e-06 | 100.000% | 0 | 2.2e-06 |
| len600 | cpu | prefill | 600 | 5.86e-03 | 8.59e-05 | 1.000000 | 8.0e-08 | 6.5e-07 | 100.000% | 0 | 2.6e-08 |
| len600 | cuda | prefill | 600 | 1.09e-02 | 2.10e-04 | 1.000000 | 3.7e-07 | 4.5e-06 | 100.000% | 0 | 3.0e-07 |
| len2k | cpu | prefill | 2048 | 5.45e-02 | 1.03e-04 | 0.999999 | 9.1e-08 | 5.4e-06 | 100.000% | 0 | 2.9e-08 |
| len2k | cuda | prefill | 2048 | 1.01e-01 | 2.31e-04 | 0.999995 | 4.4e-07 | 1.2e-04 | 100.000% | 0 | 3.6e-07 |
| len4k | cpu | prefill | 4000 | 2.12e-01 | 1.68e-04 | 0.999987 | 1.6e-07 | 4.4e-05 | 99.975% | 0 | 4.4e-08 |
| len4k | cuda | prefill | 4000 | 2.12e-01 | 2.89e-04 | 0.999985 | 6.0e-07 | 4.8e-04 | 99.950% | 0 | 5.3e-07 |
| len2k | cuda | decode | 2048 | 5.45e-02 | 1.03e-04 | 0.999999 | 8.9e-08 | 5.3e-06 | 100.000% | 0 | - |
| len2k | cuda | chunk | 2048 | 1.03e-01 | 2.31e-04 | 0.999995 | 4.5e-07 | 1.2e-04 | 100.000% | 0 | - |
| len2k | cuda | denseswa | 2048 | 5.57e-02 | 2.22e-04 | 0.999999 | 3.9e-07 | 6.6e-06 | 100.000% | 0 | - |

## Reference manifest

```json
{
 "model": "Idiap/gated-deltanet-nsa-0.4B-10B",
 "checkpoint_revision": "c283b044b5688da94e46ebbd2c50078c2c9b11ef",
 "checkpoint_sha256": "409ac2b3ff67b9b529e85be1f7f2a59c28ab8f03f78777caf379d009e3c27986",
 "gguf": "<workdir>/gguf/gdn-nsa-0.4B-f32.gguf",
 "gguf_sha256": "53afd2459dd19b05f4afa19ad83e0c841d5a1233788bd9193278a51a6ea5e272",
 "reference_repo": "idiap/hybrid-linear-sparse-attention",
 "reference_commit": "b786cbc625c0427a5becaa4ff11d9f73ecbf42eb",
 "reference_dirty": false,
 "torch": "2.7.0+cu128",
 "transformers": "4.49.0",
 "triton": "3.3.0",
 "flash_attn": "2.8.3",
 "numpy": "2.4.6",
 "reference_dtype": "float32 weights; kernel path casts q/k/v to fp16 (author's code), fp32 path has no casts",
 "reference_tf32": "disabled (matmul and cudnn)",
 "sampling": "none (greedy)",
 "dropout": "eval mode",
 "port_commit": "2ed826df0c0eefc7135e9c3acc45aaa277f9b6ff",
 "port_uncommitted_changes": true,
 "gpu": "NVIDIA B200, 595.58.03",
 "cpu": "INTEL(R) XEON(R) PLATINUM 8570",
 "host_python": "3.11.18",
 "compiler": "g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
}
```
