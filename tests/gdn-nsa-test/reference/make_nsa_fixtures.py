# Level-1 NSA fixtures: identical inputs + fla reference outputs, branch by branch, for tests/gdn-nsa-test/cpp/nsa-fixture-test.cpp.
#   <reference venv python> tests/gdn-nsa-test/reference/make_nsa_fixtures.py [out_dir]   (default fixtures/nsa)
# Per case <name>: <name>.json header + raw little-endian <name>.<tensor>.bin (row-major numpy = ggml [inner..outer]).
#   inputs : q f32 [T,32,64] (post-RoPE), k/v f32 [T,2,64], g f32 [T,32,3] gate logits (cmp, slc, swa)
#   refs   : kcmp/vcmp f32 [NBc,2,64] (fla mean_pooling, NBc = ceil(T/64)), scores f32 [T,2,NBc] (fla topk rule, f64),
#            idx i32 [T,2,16] (fla naive_nsa_topk, f64), margin f32 [T,2] (16th - 17th candidate score, inf if <= 16),
#            o_cmp/o_slc/o_swa f32 [T,32,64], o f32 [T,32,64] (fla naive_nsa with sigmoid gates)
#            o_par f32 [T,32,64] (fla Triton parallel_nsa, fp16 inputs; production reference, informational)
# Reference precision: fla's naive code in fp32 on CUDA with TF32 off (naive_nsa_sel always casts to fp32,
# naive_nsa_cmp uses flex_attention, mean_pooling is Triton); naive_nsa_topk runs in f64 on CPU. Each branch is
# cross-checked against an independent f64 masked-attention recomputation; the diffs go into the header.
# The window branch: naive_nsa calls flash_attn_func (fp16/bf16 only); it is replaced by common.sdpa_swa
# (same mask: p-(W-1) <= j <= p) run in fp32, the same substitution test_ops.py/run_ref.py use.
import json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
import torch
import fla.ops.nsa.naive as nv
from fla.ops.utils.pooling import mean_pooling
from common import sdpa_swa

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
nv.flash_attn_func = sdpa_swa

HQ, HKV, D, BS, S, W = 32, 2, 64, 64, 16, 512
G = HQ // HKV
SCALE = D ** -0.5
dev = 'cuda'


def masked_attn64(q, k, v, allowed):
    """f64 reference. q [T,HQ,D], k/v [N,HKV,D], allowed bool [T,HKV,N] -> [T,HQ,Dv]; no allowed key -> 0."""
    q, k, v = q.double().cpu(), k.double().cpu(), v.double().cpu()
    T = q.shape[0]
    o = torch.zeros(T, HQ, v.shape[-1], dtype=torch.float64)
    for t0 in range(0, T, 256):
        t1 = min(T, t0 + 256)
        for h in range(HKV):
            qh = q[t0:t1, h * G:(h + 1) * G]                       # [t, G, D]
            s = torch.einsum('tgd,nd->tgn', qh, k[:, h]) * SCALE   # [t, G, N]
            m = allowed[t0:t1, h].cpu()[:, None, :]
            s = s.masked_fill(~m, float('-inf'))
            p = torch.softmax(s, -1).nan_to_num(0.0)
            o[t0:t1, h * G:(h + 1) * G] = torch.einsum('tgn,nd->tgd', p, v[:, h])
    return o


def topk_scores64(q, k_cmp, T):
    """the naive_nsa_topk scores (before topk) in f64: [T, HKV, NBc]"""
    q, kc = q.double().cpu(), k_cmp.double().cpu()
    NBc = kc.shape[0]
    kr = kc.repeat_interleave(G, dim=1)
    logits = torch.einsum('thd,shd->ths', q, kr) * SCALE
    logits = logits.reshape(T, HKV, G, NBc)
    t = torch.arange(T)[:, None]
    s = torch.arange(NBc)[None, :]
    base = (s + 1) * BS - 1 <= t
    cur = t // BS
    forced = (s == cur) | (s == 0) | (s == cur - 1)
    logits = logits.masked_fill(~base[:, None, None, :], float('-inf'))
    sc = torch.softmax(logits, -1).nan_to_num(0.0).mean(2)
    sc = torch.where(forced[:, None, :], 1.0, sc)
    cand = base | forced
    return sc, cand


def make_inputs(T, seed, det):
    g = torch.Generator().manual_seed(seed)
    if not det:
        q = torch.randn(T, HQ, D, generator=g)
        k = torch.randn(T, HKV, D, generator=g)
        v = torch.randn(T, HKV, D, generator=g)
    else:
        # well separated block scores: q of group h = 8*u_h + small noise, keys of block b = c_{h,b}*u_h + small noise,
        # with c_{h,b} a permutation of 0, 0.5, 1, ... -> scale*q.k_cmp ~ c_{h,b} (logit gap 0.5 between neighbours)
        u = torch.randn(HKV, D, generator=g)
        u = u / u.norm(dim=-1, keepdim=True)
        NB = (T + BS - 1) // BS
        c = torch.stack([torch.randperm(NB, generator=g).float() * 0.5 for _ in range(HKV)])  # [HKV, NB]
        q = 8.0 * u.repeat_interleave(G, 0)[None] + 0.05 * torch.randn(T, HQ, D, generator=g)
        cb = c[:, torch.arange(T) // BS].T                                                   # [T, HKV]
        k = cb[..., None] * u[None] + 0.1 * torch.randn(T, HKV, D, generator=g)
        v = torch.randn(T, HKV, D, generator=g)
    gl = torch.randn(T, HQ, 3, generator=g) * 2.0
    return q, k, v, gl


def stats(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).abs().max()) if a.numel() else 0.0


def make_case(name, T, seed, det, out):
    q, k, v, gl = make_inputs(T, seed, det)
    qc, kc_, vc_ = (x[None].to(dev) for x in (q, k, v))
    gs = gl.sigmoid()
    g_cmp, g_slc, g_swa = (x[None].to(dev) for x in gs.unbind(-1))

    with torch.no_grad():
        # block means (fla mean_pooling, Triton fp32)
        k_cmp = mean_pooling(kc_, BS, None)[0]      # [NBc, HKV, D]
        v_cmp = mean_pooling(vc_, BS, None)[0]
        NBc = k_cmp.shape[0]
        # selection: fla naive_nsa_topk in f64 on CPU (on fla's k_cmp)
        idx = nv.naive_nsa_topk(q[None].double(), k_cmp[None].double().cpu(), S, BS, SCALE)[0].to(torch.int32)  # [T,HKV,S]
        idx32 = nv.naive_nsa_topk(qc, k_cmp[None], S, BS, SCALE)[0].to(torch.int32).cpu()
        sc, cand = topk_scores64(q, k_cmp, T)
        srt = sc.masked_fill(~cand[:, None, :], float('-inf')).sort(-1, descending=True).values
        if NBc > S:
            margin = (srt[..., S - 1] - srt[..., S]).float()
            margin = torch.where(torch.isinf(srt[..., S]), torch.full_like(margin, float('inf')), margin)
        else:
            margin = torch.full((T, HKV), float('inf'))
        # branches with fla's naive code (fp32)
        o_cmp, _ = nv.naive_nsa_cmp(qc, k_cmp[None], v_cmp[None], BS, SCALE, None)
        o_slc = nv.naive_nsa_sel(qc, kc_, vc_, idx[None].long().to(dev), BS, SCALE, None)
        o_swa = sdpa_swa(qc, kc_, vc_, causal=True, window_size=(W - 1, 0))
        o = nv.naive_nsa(qc, kc_, vc_, g_cmp, g_slc, g_swa, block_counts=S, block_size=BS, window_size=W, scale=SCALE)
        o_cmp, o_slc, o_swa, o = (x[0].float().cpu() for x in (o_cmp, o_slc, o_swa, o))
        o_par = None
        try:
            from fla.ops.nsa.parallel import parallel_nsa
            o_par = parallel_nsa(qc.half(), kc_.half(), vc_.half(), g_cmp, g_slc, g_swa, block_counts=S, block_size=BS,
                                 window_size=W, scale=SCALE)[0].float().cpu()
        except Exception as e:  # noqa
            print(f'  parallel_nsa failed: {e}')

    # independent f64 cross-checks
    tt = torch.arange(T)
    jj = torch.arange(T)
    causal = jj[None, :] <= tt[:, None]
    blk = jj // BS
    sel_mask = torch.zeros(T, HKV, T, dtype=torch.bool)
    for h in range(HKV):
        m = torch.zeros(T, (T + BS - 1) // BS + 1, dtype=torch.bool)
        ii = idx[:, h].long().clone()
        ii[ii < 0] = m.shape[1] - 1
        m.scatter_(1, ii, True)
        sel_mask[:, h] = m[:, blk] & causal
    o_slc64 = masked_attn64(q, k, v, sel_mask)
    swa_mask = (causal & (jj[None, :] > tt[:, None] - W))[:, None, :].expand(T, HKV, T)
    o_swa64 = masked_attn64(q, k, v, swa_mask)
    kb = torch.stack([k[b * BS:(b + 1) * BS].double().mean(0) for b in range(NBc)])
    vb = torch.stack([v[b * BS:(b + 1) * BS].double().mean(0) for b in range(NBc)])
    cmp_mask = ((torch.arange(NBc)[None, :] + 1) * BS - 1 <= tt[:, None])[:, None, :].expand(T, HKV, NBc)
    o_cmp64 = masked_attn64(q, kb, vb, cmp_mask)
    gs64 = gs.double()
    o64 = gs64[..., 0:1] * o_cmp64 + gs64[..., 1:2] * o_slc64 + gs64[..., 2:3] * o_swa64

    xc = {
        'kcmp_f64_maxabs': stats(k_cmp, kb), 'vcmp_f64_maxabs': stats(v_cmp, vb),
        'idx_topk_f32_vs_f64_set_match': float((idx32.long().sort(-1).values == idx.long().sort(-1).values).all(-1).float().mean()),
        'o_cmp_f64_maxabs': stats(o_cmp, o_cmp64), 'o_slc_f64_maxabs': stats(o_slc, o_slc64),
        'o_swa_f64_maxabs': stats(o_swa, o_swa64), 'o_f64_maxabs': stats(o, o64),
        'o_par_vs_naive_maxabs': stats(o_par, o) if o_par is not None else None,
    }
    fm = margin[torch.isfinite(margin)]
    hdr = {
        'name': name, 'T': T, 'pos0': 0, 'Hq': HQ, 'Hkv': HKV, 'D': D, 'Dv': D, 'block_size': BS, 'n_sel': S,
        'window': W, 'scale': SCALE, 'NBc': NBc, 'seed': seed, 'deterministic': int(det),
        'margin_min': float(fm.min()) if fm.numel() else None,
        'margin_rel_min': float((fm / srt[..., S - 1][torch.isfinite(margin)].float()).min()) if fm.numel() else None,
        'has_o_par': int(o_par is not None), 'crosscheck': xc,
        'files': {'q': 'f32 [T,Hq,D]', 'k': 'f32 [T,Hkv,D]', 'v': 'f32 [T,Hkv,Dv]', 'g': 'f32 [T,Hq,3] logits',
                  'kcmp': 'f32 [NBc,Hkv,D]', 'vcmp': 'f32 [NBc,Hkv,Dv]', 'scores': 'f32 [T,Hkv,NBc]',
                  'idx': 'i32 [T,Hkv,S]', 'margin': 'f32 [T,Hkv]', 'o_cmp': 'f32 [T,Hq,Dv]', 'o_slc': 'f32 [T,Hq,Dv]',
                  'o_swa': 'f32 [T,Hq,Dv]', 'o': 'f32 [T,Hq,Dv]', 'o_par': 'f32 [T,Hq,Dv]'},
    }
    arrs = {'q': q, 'k': k, 'v': v, 'g': gl, 'kcmp': k_cmp.cpu(), 'vcmp': v_cmp.cpu(), 'scores': sc.float(),
            'idx': idx, 'margin': margin, 'o_cmp': o_cmp, 'o_slc': o_slc, 'o_swa': o_swa, 'o': o}
    if o_par is not None:
        arrs['o_par'] = o_par
    for key, a in arrs.items():
        a = a.contiguous().numpy()
        a = a.astype('<i4') if a.dtype == np.int32 else a.astype('<f4')
        a.tofile(os.path.join(out, f'{name}.{key}.bin'))
    with open(os.path.join(out, f'{name}.json'), 'w') as f:
        json.dump(hdr, f, indent=1)
    print(f'{name}: T={T} NBc={NBc} margin_min={hdr["margin_min"]} xc={xc}', flush=True)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'fixtures/nsa'
    os.makedirs(out, exist_ok=True)
    cases = [(f'rand_T{T}', T, 1000 + T, False) for T in (1, 2, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1000, 2048)]
    cases.append(('det_T2048', 2048, 7, True))
    for c in cases:
        make_case(*c, out)
    with open(os.path.join(out, 'cases.txt'), 'w') as f:
        f.write('\n'.join(c[0] for c in cases) + '\n')


if __name__ == '__main__':
    main()
