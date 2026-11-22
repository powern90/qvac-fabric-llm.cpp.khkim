ggml_cuda_init: found 1 CUDA devices (Total VRAM: 182624 MiB):
  Device 0: NVIDIA B200, compute capability 10.0, VMM: yes, VRAM: 182624 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |  pp2048 @ d3072 |    26942.82 ± 964.35 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |   tg128 @ d3072 |        376.98 ± 5.36 |

build: 2ed826df0 (9246)
