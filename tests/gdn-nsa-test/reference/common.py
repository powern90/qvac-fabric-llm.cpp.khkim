# Shared helpers: fp32 SDPA sliding window, fp32 recurrent gated delta rule, source-level monkeypatching.
import inspect, textwrap
import torch
import torch.nn.functional as F


def sdpa_swa(q, k, v, causal=True, window_size=(511, 0), **unused):
    """Drop-in for flash_attn_func(q,k,v,causal=True,window_size=(W-1,0)) in float32.
    q [B,TQ,HQ,D], k/v [B,TK,H,D]; query i is at absolute position (TK-TQ)+i (flash-attn bottom-right alignment).
    key j visible iff pos_i - (W-1) <= j <= pos_i."""
    assert causal and window_size[1] == 0
    B, TQ, HQ, D = q.shape
    TK, H = k.shape[1], k.shape[2]
    qi = torch.arange(TQ, device=q.device)[:, None] + (TK - TQ)
    kj = torch.arange(TK, device=q.device)[None, :]
    mask = (kj <= qi) & (kj >= qi - window_size[0])
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       attn_mask=mask, enable_gqa=(HQ != H))
    return o.transpose(1, 2).contiguous()


def l2norm_ref(x, eps=1e-6):
    # same formula as fla.modules.l2norm.l2norm_fwd (used by use_qk_l2norm_in_kernel=True)
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_delta_rule_ref(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                         cu_seqlens=None, use_qk_l2norm_in_kernel=False, **unused):
    """Float32 recurrent gated delta rule (same math as tests/ops/test_gated_delta.py::recurrent_gated_delta_rule_ref).
    q,k [B,T,H,K], v [B,T,H,V], g (log decay) [B,T,H], beta [B,T,H]. Returns o [B,T,H,V], state [B,H,K,V]."""
    assert cu_seqlens is None
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = l2norm_ref(q), l2norm_ref(k)
    B, T, H, K = k.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q * scale
    h = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32) if initial_state is None else initial_state.float().clone()
    o = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    for t in range(T):
        h = h * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        vt = (v[:, t] - torch.einsum('bhk,bhkv->bhv', kt, h)) * beta[:, t][..., None]
        h = h + kt[..., :, None] * vt[..., None, :]
        o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t], h)
    return o, (h if output_final_state else None)


def _div(a, b):
    # true fp32 division: CUDA torch computes a / <python scalar> as a * (1/b)
    return a / torch.full_like(a, b)


def kv_fake_quant(x, qtype):
    """Quantize -> dequantize fp32 x along the last dim in blocks of 32, bit-exact with ggml
    quantize_row_{q8_0,q4_0}_ref + dequantize_row_{q8_0,q4_0} (scale d is stored as fp16)."""
    assert x.dtype == torch.float32 and x.shape[-1] % 32 == 0
    b = x.reshape(-1, 32)
    if qtype == 'q8_0':
        d = _div(b.abs().amax(-1, keepdim=True), 127.0)
        id = torch.where(d == 0, torch.zeros_like(d), torch.reciprocal(d))
        x0 = b * id
        a = x0.abs()
        f = a.floor()
        q = (torch.where(a - f >= 0.5, f + 1, f) * x0.sign()).to(torch.int8).float()  # roundf, stored as int8
    elif qtype == 'q4_0':
        a = b.abs()
        imax = a.argmax(-1, keepdim=True)                            # first index of max |x|
        mx = torch.where(a.gather(-1, imax) == 0, torch.zeros_like(b[:, :1]), b.gather(-1, imax))
        d = _div(mx, -8.0)
        id = torch.where(d == 0, torch.zeros_like(d), torch.reciprocal(d))
        x0 = b * id
        q = (x0 + 8.5).trunc().clamp(max=15) - 8                     # MIN(15, (int8_t)(x0 + 8.5f)) - 8
    else:
        raise ValueError(qtype)
    return (q * d.half().float()).reshape(x.shape)


def patch_method(cls, name, replacements, extra_globals=None):
    """Re-define cls.name from its source with exact string replacements (each must occur)."""
    fn = getattr(cls, name)
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new in replacements:
        assert src.count(old) >= 1, f'pattern not found in {cls.__name__}.{name}: {old!r}'
        src = src.replace(old, new)
    g = dict(fn.__globals__)
    if extra_globals:
        g.update(extra_globals)
    ns = {}
    src = 'from __future__ import annotations\n' + src  # source modules use postponed annotations
    exec(compile(src, f'<patched {cls.__name__}.{name}>', 'exec'), g, ns)
    orig = fn
    setattr(cls, name, ns[name])
    return orig
