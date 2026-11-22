# Verify the prebuilt flash-attn wheel runs on this GPU and matches an SDPA reference
# for the exact call used by NSA: flash_attn_func(q,k,v,causal=True,window_size=(W-1,0)).
import torch, torch.nn.functional as F
from flash_attn import flash_attn_func
torch.manual_seed(0)
W = 512

def sdpa_swa(q, k, v, window_size):
    # q [B,TQ,HQ,D], k/v [B,TK,H,D]; query i sits at absolute position (TK-TQ)+i
    B, TQ, HQ, D = q.shape
    TK, H = k.shape[1], k.shape[2]
    qi = torch.arange(TQ, device=q.device)[:, None] + (TK - TQ)
    kj = torch.arange(TK, device=q.device)[None, :]
    mask = (kj <= qi) & (kj >= qi - (window_size - 1))
    o = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
                                       attn_mask=mask, enable_gqa=(HQ != H))
    return o.transpose(1, 2)

for TQ, TK in [(2048, 2048), (4000, 4000), (1, 2100), (1, 300), (7, 7)]:
    q = torch.randn(1, TQ, 32, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, TK, 2, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, TK, 2, 64, device='cuda', dtype=torch.float16)
    o_fa = flash_attn_func(q, k, v, causal=True, window_size=(W - 1, 0)).float()
    o_ref = sdpa_swa(q, k, v, W)
    print(f"TQ={TQ} TK={TK} max|fa-sdpa_fp32|={(o_fa - o_ref).abs().max().item():.3e}")
