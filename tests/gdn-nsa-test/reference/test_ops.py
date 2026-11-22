# Unit sanity checks: naive_nsa (fp32, SDPA window) vs parallel_nsa (fp16 kernels); GDN ref vs chunk kernel.
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import torch
import fla.ops.nsa.naive as nv
from fla.ops.nsa.parallel import parallel_nsa
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from common import sdpa_swa, gated_delta_rule_ref
nv.flash_attn_func = sdpa_swa
torch.manual_seed(0)
T = 700
q = torch.randn(1, T, 32, 64, device='cuda')
k = torch.randn(1, T, 2, 64, device='cuda')
v = torch.randn(1, T, 2, 64, device='cuda')
g = torch.rand(1, T, 32, 3, device='cuda')
gc, gs, gw = g.unbind(-1)
o_k = parallel_nsa(q.half(), k.half(), v.half(), gc, gs, gw, block_counts=16, block_size=64, window_size=512).float()
o_n, bi = nv.naive_nsa(q, k, v, gc, gs, gw, block_counts=16, block_size=64, window_size=512, return_block_indices=True)
print('nsa max|kernel-naive|', (o_k - o_n).abs().max().item(), 'mean', (o_k - o_n).abs().mean().item())
from fla.ops.nsa.parallel import parallel_nsa_topk
from fla.ops.nsa.compression import parallel_nsa_compression
from fla.ops.utils.pooling import mean_pooling
kc = mean_pooling(k.half(), 64, None)
_, lse = parallel_nsa_compression(q=q.half(), k=kc, v=mean_pooling(v.half(), 64, None), TK=T, block_size=64, scale=64**-0.5, cu_seqlens=None)
bk = parallel_nsa_topk(q=q.half(), k=kc, lse=lse, TK=T, block_counts=16, block_size=64, scale=64**-0.5)
sk = bk.long().sort(-1).values; sn = bi.long().sort(-1).values
print('block set agreement kernel vs naive', (sk == sn).all(-1).float().mean().item())
# GDN
B, H, K = 1, 8, 128
qq = torch.randn(B, T, H, K, device='cuda'); kk = torch.randn(B, T, H, K, device='cuda'); vv = torch.randn(B, T, H, K, device='cuda')
gg = -torch.rand(B, T, H, device='cuda') * 0.5; bb = torch.rand(B, T, H, device='cuda')
o1, s1 = chunk_gated_delta_rule(qq.half(), kk.half(), vv.half(), gg, bb, output_final_state=True, use_qk_l2norm_in_kernel=True)
# note: chunk_gated_delta_rule asserts q.dtype != float32 (fla/ops/gated_delta_rule/chunk.py:300)
o2, s2 = gated_delta_rule_ref(qq, kk, vv, gg, bb, output_final_state=True, use_qk_l2norm_in_kernel=True)
print('gdn max|chunk16-ref|', (o1.float() - o2).abs().max().item(), 'state max diff', (s1.float()-s2).abs().max().item(), s1.dtype)
