| run | vs | T | max abs | mean abs | KL mean | KL max | top-1 | block sets equal | mismatches | max margin of a mismatch |
|---|---|---|---|---|---|---|---|---|---|---|
| len2k_cpu_prefill | kernel | 2048 | 5.45e-02 | 3.48e-04 | 9.1e-08 | 5.4e-06 | 100.000% | 99.919% | 40 | 3.3e-05 |
| len2k_cpu_prefill | fp32 | 2048 | 5.45e-02 | 1.79e-04 | 2.9e-08 | 4.8e-06 | 100.000% | 99.984% | 8 | - |
| len2k_cuda_chunk | kernel | 2048 | 1.03e-01 | 8.36e-04 | 4.5e-07 | 1.2e-04 | 100.000% | 99.890% | 54 | 3.3e-05 |
| len2k_cuda_chunk | fp32 | 2048 | 1.03e-01 | 7.65e-04 | 3.7e-07 | 1.2e-04 | 100.000% | 99.955% | 22 | - |
| len2k_cuda_decode | kernel | 2048 | 5.45e-02 | 3.47e-04 | 8.9e-08 | 5.3e-06 | 100.000% | 99.917% | 41 | 3.7e-05 |
| len2k_cuda_decode | fp32 | 2048 | 5.45e-02 | 1.78e-04 | 2.8e-08 | 4.6e-06 | 100.000% | 99.986% | 7 | - |
| len2k_cuda_denseswa | kernel | 2048 | 5.57e-02 | 8.41e-04 | 3.9e-07 | 6.6e-06 | 100.000% | 99.910% | 44 | 4.0e-05 |
| len2k_cuda_denseswa | fp32 | 2048 | 5.57e-02 | 7.70e-04 | 3.1e-07 | 7.0e-06 | 100.000% | 99.967% | 16 | - |
| len2k_cuda_prefill | kernel | 2048 | 1.01e-01 | 8.36e-04 | 4.4e-07 | 1.2e-04 | 100.000% | 99.896% | 51 | 3.3e-05 |
| len2k_cuda_prefill | fp32 | 2048 | 1.01e-01 | 7.65e-04 | 3.6e-07 | 1.2e-04 | 100.000% | 99.965% | 17 | - |
| len4k_cpu_prefill | kernel | 4000 | 2.12e-01 | 4.18e-04 | 1.6e-07 | 4.4e-05 | 99.975% | 99.772% | 219 | 8.7e-05 |
| len4k_cpu_prefill | fp32 | 4000 | 5.45e-02 | 1.95e-04 | 4.4e-08 | 3.2e-05 | 100.000% | 99.969% | 30 | - |
| len4k_cuda_prefill | kernel | 4000 | 2.12e-01 | 8.98e-04 | 6.0e-07 | 4.8e-04 | 99.950% | 99.744% | 246 | 9.6e-05 |
| len4k_cuda_prefill | fp32 | 4000 | 1.28e-01 | 8.13e-04 | 5.3e-07 | 4.9e-04 | 99.975% | 99.892% | 104 | - |
| len600_cpu_prefill | kernel | 600 | 5.86e-03 | 3.25e-04 | 8.0e-08 | 6.5e-07 | 100.000% | 100.000% | 0 | - |
| len600_cpu_prefill | fp32 | 600 | 3.18e-03 | 1.71e-04 | 2.6e-08 | 4.8e-07 | 100.000% | 100.000% | 0 | - |
| len600_cuda_prefill | kernel | 600 | 1.09e-02 | 8.21e-04 | 3.7e-07 | 4.5e-06 | 100.000% | 100.000% | 0 | - |
| len600_cuda_prefill | fp32 | 600 | 9.81e-03 | 7.58e-04 | 3.0e-07 | 4.0e-06 | 100.000% | 100.000% | 0 | - |
| short_cpu_prefill | kernel | 6 | 2.67e-03 | 1.65e-04 | 1.7e-08 | 2.9e-08 | 100.000% | 100.000% | 0 | - |
| short_cpu_prefill | fp32 | 6 | 1.62e-03 | 1.10e-04 | 1.1e-08 | 4.0e-08 | 100.000% | 100.000% | 0 | - |
| short_cuda_prefill | kernel | 6 | 1.45e-02 | 2.42e-03 | 2.4e-06 | 7.7e-06 | 100.000% | 100.000% | 0 | - |
| short_cuda_prefill | fp32 | 6 | 1.48e-02 | 2.41e-03 | 2.2e-06 | 7.0e-06 | 100.000% | 100.000% | 0 | - |

- greedy short: llama.cpp CUDA vs reference (nocache): first 32/32 tokens identical
- greedy short: llama.cpp CUDA vs reference (cache): first 32/32 tokens identical
- greedy len2k: llama.cpp CUDA vs reference (nocache): first 32/32 tokens identical
- greedy len2k: llama.cpp CUDA vs reference (cache): first 32/32 tokens identical
- len2k cuda_decode vs cuda prefill: max abs 1.01e-01, top-1 100.00%
- len2k cuda_chunk vs cuda prefill: max abs 2.53e-02, top-1 100.00%
- len2k cuda_denseswa vs cuda prefill: max abs 1.01e-01, top-1 100.00%

- perplexity len2k: ref kernel 19.4027, ref fp32 19.4031, llama.cpp CPU 19.4030, llama.cpp CUDA 19.4033
- perplexity len4k: ref kernel 14.6317, ref fp32 14.6317, llama.cpp CPU 14.6316, llama.cpp CUDA 14.6317

| layer | type | rel. err CPU | rel. err CUDA |
|---|---|---|---|
| 0 | GDN | 1.1e-04 | 1.5e-04 |
| 1 | NSA | 1.3e-04 | 2.5e-04 |
| 2 | GDN | 1.5e-04 | 2.8e-04 |
| 3 | NSA | 1.5e-04 | 3.1e-04 |
| 4 | GDN | 1.7e-04 | 3.3e-04 |
| 5 | NSA | 1.7e-04 | 3.5e-04 |
| 6 | GDN | 1.9e-04 | 3.7e-04 |
| 7 | NSA | 1.8e-04 | 3.8e-04 |
| 8 | GDN | 2.0e-04 | 3.9e-04 |
| 9 | NSA | 2.3e-04 | 4.2e-04 |
| 10 | GDN | 2.3e-04 | 4.3e-04 |
| 11 | NSA | 2.5e-04 | 4.6e-04 |
| 12 | GDN | 2.6e-04 | 4.7e-04 |
| 13 | NSA | 2.6e-04 | 4.9e-04 |
| 14 | GDN | 2.6e-04 | 4.8e-04 |
| 15 | NSA | 2.8e-04 | 5.1e-04 |
| 16 | GDN | 2.7e-04 | 4.8e-04 |
| 17 | NSA | 3.0e-04 | 6.1e-04 |
| 18 | GDN | 2.9e-04 | 5.8e-04 |
| 19 | NSA | 2.9e-04 | 5.7e-04 |
| 20 | GDN | 2.8e-04 | 5.6e-04 |
| 21 | NSA | 3.3e-04 | 5.8e-04 |
| 22 | GDN | 3.1e-04 | 5.7e-04 |
| 23 | NSA | 2.9e-04 | 5.3e-04 |
