ggml_cuda_init: found 1 CUDA devices (Total VRAM: 182624 MiB):
  Device 0: NVIDIA B200, compute capability 10.0, VMM: yes, VRAM: 182624 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |           pp512 |  42668.46 ± 11789.32 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |          pp2048 |   39918.52 ± 2184.22 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |           tg128 |        459.71 ± 3.03 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |   pp512 @ d1024 |   35713.98 ± 5524.81 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |  pp2048 @ d1024 |   35895.26 ± 3491.98 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |   tg128 @ d1024 |        431.43 ± 3.23 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |   pp512 @ d3072 |   35711.56 ± 5760.10 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |  pp2048 @ d3072 |   37138.73 ± 1693.02 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |   tg128 @ d3072 |        401.91 ± 2.84 |

build: 2ed826df0 (9246)
