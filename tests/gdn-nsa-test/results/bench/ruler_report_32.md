# RULER S-NIAH subset (N = 20 per task and length, our generator with RULER templates)

Not a reproduction of the paper: own sample generator (RULER niah.py logic and RULER strings), greedy decoding with max_new_tokens = 32 (RULER: 128), RULER string_match_all scoring. Paper = laNSA 0.4B, Table 2. FLA = HF/fla reference (author kernel path, fp32 weights) on GPU; ggml = gdn-nsa-0.4B-f32.gguf, F16 KV cache. Delta = ggml - FLA in pp on the same samples. agree = identical generated token sequence (tok) / same per-sample correctness (corr).

| Task | Paper | FLA | ggml CPU | ggml GPU | Δ CPU | Δ GPU | agree CPU | agree GPU |
|---|---|---|---|---|---|---|---|---|
| S1 1K | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| S1 2K | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| S1 4K | 65.0 | 55.0 | 55.0 | 55.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| S2 1K | 88.6 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| S2 2K | 100.0 | 75.0 | 75.0 | 75.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 19/20 tok, 20/20 corr |
| S2 4K | 37.6 | 30.0 | 30.0 | 30.0 | +0.0 | +0.0 | 18/20 tok, 20/20 corr | 18/20 tok, 20/20 corr |
| S3 1K | 97.6 | 20.0 | 20.0 | 20.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| S3 2K | 73.0 | 0.0 | 0.0 | 0.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| S3 4K | 11.8 | 0.0 | 0.0 | 0.0 | +0.0 | +0.0 | 20/20 tok, 20/20 corr | 20/20 tok, 20/20 corr |
| **Average** | 74.8 | 53.3 | 53.3 | 53.3 | +0.0 | +0.0 | 178/180 tok, 180/180 corr | 177/180 tok, 180/180 corr |

Token-sequence disagreements FLA vs ggml CPU: 2:
- S2_4096_07: first differing step 5, score FLA 0 / ggml 0; FLA '1.0. What is the special magic number for light-surplus ment' vs ggml '1.0. The special magic number for light-surplus mentioned in'
- S2_4096_10: first differing step 9, score FLA 1 / ggml 1; FLA '5107245.\nWhat is the special magic number for yellow-step-si' vs ggml '5107245. What is the special magic number for the number of '

Token-sequence disagreements FLA vs ggml GPU: 3:
- S2_2048_02: first differing step 11, score FLA 1 / ggml 1; FLA ': 4823498.\n[1] "The Internet is the most powerful tool ever ' vs ggml ': 4823498.\nWhat is the special magic number for animated-por'
- S2_4096_07: first differing step 5, score FLA 0 / ggml 0; FLA '1.0. What is the special magic number for light-surplus ment' vs ggml '1.0. The special magic number for light-surplus mentioned in'
- S2_4096_10: first differing step 9, score FLA 1 / ggml 1; FLA '5107245.\nWhat is the special magic number for yellow-step-si' vs ggml '5107245. What is the special magic number for the number of '

FLA worker overlap: 17 samples generated twice, 0 differ.
