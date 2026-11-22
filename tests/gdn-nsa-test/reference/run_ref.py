#!/usr/bin/env python
"""Golden reference outputs for Idiap/gated-deltanet-nsa-0.4B-10B (hybrid Gated DeltaNet + NSA).

Stages (run from the work dir, GDN_NSA_CKPT=<hf checkpoint dir>):
  kernel  : author code path (fp32 weights, layers cast q/k/v to fp16, Triton kernels + flash-attn SWA)
            -> logits_NAME.npy, blockidx_NAME.npy, margin_NAME.npy, hidden_len2k.npy
  fp32    : pure-PyTorch fp32 reference (no fp16 casts; naive_nsa + SDPA window; recurrent GDN in fp32)
            -> logits_NAME_fp32.npy, blockidx_NAME_fp32.npy
            --kv-fake-quant q8_0|q4_0: NSA k/v go through ggml quantize->dequantize (as a llama.cpp -ctk/-ctv cache)
            -> logits_NAME_fp32_kvq8_0.npy, blockidx_NAME_fp32_kvq8_0.npy (info_fp32_kvq8_0.json)
  greedy  : 32 greedy tokens, no-cache repeated forward vs model.generate(use_cache=True)
  compare : kernel vs fp32 metrics (logits, block sets) -> printed and stored under "compare" in info_compare.json
"""
import argparse, faulthandler, json, os, sys, time, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# outputs (and tokens_*.txt) live in the golden dir
HERE = os.environ.get('GDN_NSA_GOLDEN', 'golden')
warnings.filterwarnings('ignore')

import numpy as np
import torch

CKPT = os.environ.get('GDN_NSA_CKPT', 'ckpt')
PROMPTS = ['short', 'len600', 'len2k', 'len4k']
BS, S = 64, 16


def load_tokens(name):
    with open(os.path.join(HERE, f'tokens_{name}.txt')) as f:
        return [int(x) for x in f.read().split()]


def load_model():
    import fla  # noqa: F401  (registers hybrid_gated_deltanet with transformers)
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).cuda().eval()
    return model


def sort_blocks(bi):
    """[..., S] int -> ascending order with -1 padding at the end (selection order is irrelevant)."""
    bi = bi.long()
    big = torch.iinfo(torch.long).max
    s = torch.where(bi < 0, big, bi).sort(-1).values
    return torch.where(s == big, -1, s).to(torch.int32)


@torch.no_grad()
def kernel_scores(q, k_cmp, lse, TK, block_size=BS):
    """Re-derive the importance scores exactly as parallel_nsa_kernel_topk does, in torch.
    q [1,TQ,HQ,D] fp16, k_cmp [1,TC,H,D] fp16, lse [1,TQ,HQ] fp32 (from parallel_nsa_compression).
    Returns score [TQ,H,TC] (mean over the G q-heads of a kv group; forced blocks = 1.0; ineligible = -1)."""
    _, TQ, HQ, D = q.shape
    TC, H = k_cmp.shape[1], k_cmp.shape[2]
    G = HQ // H
    scale = D ** -0.5
    qs = (q[0].float() * scale).half().float()                      # kernel: (b_q * scale).to(fp16)
    kc = k_cmp[0].float().repeat_interleave(G, dim=1)                # [TC,HQ,D]
    s = torch.einsum('thd,chd->thc', qs, kc)                         # [TQ,HQ,TC]
    t = torch.arange(TQ, device=q.device)[:, None] + (TK - TQ)
    c = torch.arange(TC, device=q.device)[None, :]
    IC = t // block_size                                             # current query block
    valid = c < IC                                                   # blocks strictly before current block
    p = torch.where(valid[:, None, :], torch.exp(s - lse[0][..., None]), torch.zeros_like(s))
    forced = (c == 0) | (c == IC - 1) | (c == IC)
    p = torch.where(forced[:, None, :], torch.ones_like(p), p)
    score = p.view(TQ, H, G, TC).mean(2)                             # kernel sums over G; mean = sum/G (same order)
    eligible = (c <= IC)
    score = torch.where(eligible[:, None, :], score, torch.full_like(score, -1.0))
    return score, eligible


@torch.no_grad()
def margins_and_sel(score, eligible):
    TQ, H, TC = score.shape
    srt, idx = score.sort(-1, descending=True)
    n_el = eligible.sum(-1)                                          # [TQ]
    if TC > S:
        margin = srt[..., S - 1] - srt[..., S]
    else:
        margin = torch.full((TQ, H), float('inf'), device=score.device)
    margin = torch.where((n_el > S)[:, None], margin, torch.full_like(margin, float('inf')))
    k = min(S, TC)
    sel = idx[..., :k]
    sel = torch.where(srt[..., :k] >= 0, sel, torch.full_like(sel, -1))
    if k < S:
        sel = torch.cat([sel, sel.new_full((TQ, H, S - k), -1)], -1)
    return margin, sort_blocks(sel)


# --------------------------------------------------------------------------------------------- kernel
class KernelRecorder:
    """Wraps fla.layers.nsa.parallel_nsa: computes block indices with the author's own Triton topk kernel,
    records them, then calls the original parallel_nsa with those indices (identical computation)."""

    def __init__(self):
        import fla.layers.nsa as nsa_mod
        from fla.ops.nsa.parallel import parallel_nsa, parallel_nsa_topk
        from fla.ops.nsa.compression import parallel_nsa_compression
        from fla.ops.utils.pooling import mean_pooling
        self.mod, self.orig = nsa_mod, parallel_nsa
        self.topk, self.comp, self.pool = parallel_nsa_topk, parallel_nsa_compression, mean_pooling
        self.records = []

    def __call__(self, q, k, v, g_cmp=None, g_slc=None, g_swa=None, block_indices=None, block_counts=16,
                 block_size=64, window_size=0, scale=None, cu_seqlens=None):
        assert cu_seqlens is None and block_indices is None
        sc = k.shape[-1] ** -0.5
        k_cmp, v_cmp = self.pool(k, block_size, None), self.pool(v, block_size, None)
        _, lse = self.comp(q=q, k=k_cmp, v=v_cmp, TK=k.shape[1], block_size=block_size, scale=sc, cu_seqlens=None)
        bi = self.topk(q=q, k=k_cmp, lse=lse, TK=k.shape[1], block_counts=block_counts, block_size=block_size,
                       scale=sc, cu_seqlens=None)
        score, eligible = kernel_scores(q, k_cmp, lse, k.shape[1], block_size)
        margin, sel_torch = margins_and_sel(score, eligible)
        self.records.append(dict(bi=sort_blocks(bi[0]).cpu(), margin=margin.cpu(), sel_torch=sel_torch.cpu()))
        return self.orig(q=q, k=k, v=v, g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa, block_indices=bi,
                         block_counts=block_counts, block_size=block_size, window_size=window_size,
                         scale=scale, cu_seqlens=None)

    def __enter__(self):
        self.mod.parallel_nsa = self
        return self

    def __exit__(self, *a):
        self.mod.parallel_nsa = self.orig


@torch.no_grad()
def stage_kernel(model, names, info):
    for name in names:
        ids = torch.tensor([load_tokens(name)], device='cuda')
        T = ids.shape[1]
        t0 = time.time()
        logits = model(ids, use_cache=False).logits[0].float()
        torch.cuda.synchronize()
        dt = time.time() - t0
        np.save(os.path.join(HERE, f'logits_{name}.npy'), logits.cpu().numpy())
        assert torch.isfinite(logits).all(), f'non-finite logits in {name}'
        # second pass with the recorder (block indices + margins)
        with KernelRecorder() as rec:
            logits2 = model(ids, use_cache=False).logits[0].float()
        assert len(rec.records) == 12
        bi = torch.stack([r['bi'] for r in rec.records]).numpy().astype(np.int32)          # [12,T,2,16]
        mg = torch.stack([r['margin'] for r in rec.records]).numpy().astype(np.float32)     # [12,T,2]
        st = torch.stack([r['sel_torch'] for r in rec.records]).numpy().astype(np.int32)
        np.save(os.path.join(HERE, f'blockidx_{name}.npy'), bi)
        np.save(os.path.join(HERE, f'margin_{name}.npy'), mg)
        d = dict(T=T, fwd_sec=round(dt, 3),
                 recorder_pass_max_logit_diff=float((logits2 - logits).abs().max()),
                 torch_rederived_selection_set_agreement=float((st == bi).all(-1).mean()),
                 min_margin_finite=float(mg[np.isfinite(mg)].min()) if np.isfinite(mg).any() else None,
                 frac_margin_lt_1em3=float((mg < 1e-3).mean()), frac_margin_lt_1em4=float((mg < 1e-4).mean()))
        info.setdefault('kernel', {})[name] = d
        print('kernel', name, d, flush=True)

        if name == 'len2k':
            hs, hooks = [], []
            hooks.append(model.model.embeddings.register_forward_hook(lambda m, i, o: hs.append(o.detach())))
            for layer in model.model.layers:
                hooks.append(layer.register_forward_hook(lambda m, i, o: hs.append(o[0].detach())))
            out = model(ids, use_cache=False, output_hidden_states=True)
            for h in hooks:
                h.remove()
            H = torch.stack([h[0] for h in hs]).float()                                       # [25,T,1024]
            np.save(os.path.join(HERE, 'hidden_len2k.npy'), H.cpu().numpy())
            hf = out.hidden_states
            normed = model.model.norm(hs[-1])
            info['hidden_len2k'] = dict(
                n_hf_hidden_states=len(hf),
                hf_0_23_match_hooks_maxdiff=float(max((hf[i][0] - H[i]).abs().max() for i in range(24))),
                hf_last_minus_hook_last_prenorm=float((hf[24][0] - H[24]).abs().max()),
                hf_last_minus_norm_of_hook_last=float((hf[24] - normed).abs().max()),
            )
            print('hidden', info['hidden_len2k'], flush=True)


# --------------------------------------------------------------------------------------------- fp32
def install_fp32_patches(records, kv_quant=None):
    import fla.ops.nsa.naive as nv
    from fla.layers.nsa import NativeSparseAttention
    from fla.layers.gated_deltanet import GatedDeltaNet
    from common import sdpa_swa, gated_delta_rule_ref, patch_method, kv_fake_quant
    nv.flash_attn_func = sdpa_swa  # fp32 SDPA sliding window (flash-attn has no fp32)

    def nsa_fp32(q, k, v, g_cmp=None, g_slc=None, g_swa=None, block_size=64, block_counts=16, window_size=0,
                 cu_seqlens=None):
        assert q.dtype == torch.float32 and k.dtype == torch.float32 and cu_seqlens is None
        if kv_quant:
            # k is post-RoPE [B,T,H,D]; all NSA branches read the dequantized k/v
            k, v = kv_fake_quant(k, kv_quant), kv_fake_quant(v, kv_quant)
        o, bi = nv.naive_nsa(q=q, k=k, v=v, g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa, block_counts=block_counts,
                             block_size=block_size, window_size=window_size, cu_seqlens=None,
                             return_block_indices=True)
        records.append(sort_blocks(bi[0]).cpu())
        return o

    patch_method(NativeSparseAttention, 'forward', [
        ('if q.dtype == torch.float32:', 'if False:  # PATCHED: no fp16 cast'),
        ('o = parallel_nsa(', 'o = _nsa_fp32('),
    ], extra_globals={'_nsa_fp32': nsa_fp32})
    patch_method(GatedDeltaNet, 'forward', [
        ('if q.dtype == torch.float32:', 'if False:  # PATCHED: no fp16 cast'),
        ('o, recurrent_state = chunk_gated_delta_rule(', 'o, recurrent_state = _gdr_ref('),
        ('o, recurrent_state = fused_recurrent_gated_delta_rule(', 'o, recurrent_state = _gdr_ref('),
    ], extra_globals={'_gdr_ref': gated_delta_rule_ref})


@torch.no_grad()
def stage_fp32(model, names, info, kv_quant=None):
    records = []
    install_fp32_patches(records, kv_quant)
    tag = f'fp32_kv{kv_quant}' if kv_quant else 'fp32'
    for name in names:
        ids = torch.tensor([load_tokens(name)], device='cuda')
        records.clear()
        t0 = time.time()
        logits = model(ids, use_cache=False).logits[0].float()
        torch.cuda.synchronize()
        dt = time.time() - t0
        assert len(records) == 12
        assert torch.isfinite(logits).all(), f'non-finite fp32 logits in {name}'
        np.save(os.path.join(HERE, f'logits_{name}_{tag}.npy'), logits.cpu().numpy())
        np.save(os.path.join(HERE, f'blockidx_{name}_{tag}.npy'), torch.stack(records).numpy().astype(np.int32))
        info.setdefault(tag, {})[name] = dict(T=ids.shape[1], fwd_sec=round(dt, 1))
        print(tag, name, info[tag][name], flush=True)


# --------------------------------------------------------------------------------------------- greedy
@torch.no_grad()
def stage_greedy(model, names, info, n_new=32):
    for name in names:
        prompt = load_tokens(name)
        ids = torch.tensor([prompt], device='cuda')
        cur, nc_tok, nc_logits = ids, [], []
        for _ in range(n_new):
            lg = model(cur, use_cache=False).logits[0, -1].float()
            nxt = int(lg.argmax())
            nc_tok.append(nxt)
            nc_logits.append(lg)
            cur = torch.cat([cur, cur.new_tensor([[nxt]])], 1)
        out = model.generate(ids, max_new_tokens=n_new, min_new_tokens=0, do_sample=False, use_cache=True,
                             eos_token_id=None, pad_token_id=0, return_dict_in_generate=True, output_logits=True)
        c_tok = out.sequences[0, ids.shape[1]:].tolist()
        c_logits = [l[0].float() for l in out.logits]
        with open(os.path.join(HERE, f'greedy_{name}_nocache.txt'), 'w') as f:
            f.write(' '.join(map(str, nc_tok)) + '\n')
        with open(os.path.join(HERE, f'greedy_{name}_cache.txt'), 'w') as f:
            f.write(' '.join(map(str, c_tok)) + '\n')
        agree = sum(a == b for a, b in zip(nc_tok, c_tok))
        first_div = next((i for i, (a, b) in enumerate(zip(nc_tok, c_tok)) if a != b), None)
        n_cmp = first_div + 1 if first_div is not None else len(c_tok)   # logits comparable while prefixes agree
        step_diff = [float((nc_logits[i] - c_logits[i]).abs().max()) for i in range(min(n_cmp, len(c_logits)))]
        top2 = [float(torch.topk(nc_logits[i], 2).values.diff().abs()) for i in range(n_new)]
        d = dict(n_cache=len(c_tok), identical=(nc_tok == c_tok), n_agree=agree, first_divergence=first_div,
                 cache_vs_nocache_step_logit_maxdiff=step_diff, max_step_logit_diff=max(step_diff),
                 nocache_top1_top2_gap_min=min(top2))
        info.setdefault('greedy', {})[name] = d
        print('greedy', name, {k: v for k, v in d.items() if k != 'cache_vs_nocache_step_logit_maxdiff'}, flush=True)


# --------------------------------------------------------------------------------------------- compare
@torch.no_grad()
def stage_compare(names, info):
    rows = []
    for name in names:
        pk, pf = os.path.join(HERE, f'logits_{name}.npy'), os.path.join(HERE, f'logits_{name}_fp32.npy')
        if not (os.path.exists(pk) and os.path.exists(pf)):
            continue
        a = torch.from_numpy(np.load(pk)).cuda().double()      # kernel
        b = torch.from_numpy(np.load(pf)).cuda().double()      # fp32
        d = (a - b).abs()
        lp, lq = torch.log_softmax(b, -1), torch.log_softmax(a, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)                    # KL(fp32 || kernel) per position
        top1 = (a.argmax(-1) == b.argmax(-1)).double()
        r = dict(prompt=name, T=a.shape[0], max_abs=float(d.max()), mean_abs=float(d.mean()),
                 kl_mean=float(kl.mean()), kl_max=float(kl.max()), top1_pct=100 * float(top1.mean()))
        pos_max = d.max(-1).values                             # per-position max |diff|
        r.update(pos_maxabs_p50=float(pos_max.median()), pos_maxabs_p99=float(torch.quantile(pos_max, 0.99)),
                 argmax_pos_of_max_abs=int(pos_max.argmax()))
        bk, bf = os.path.join(HERE, f'blockidx_{name}.npy'), os.path.join(HERE, f'blockidx_{name}_fp32.npy')
        if os.path.exists(bk) and os.path.exists(bf):
            x, y = np.load(bk), np.load(bf)
            mg = np.load(os.path.join(HERE, f'margin_{name}.npy'))
            same = (x == y).all(-1)                            # [12,T,2]
            inter = np.array([[[len(set(x[l, t, h]) & set(y[l, t, h]) - {-1}) for h in range(2)]
                               for t in range(x.shape[1])] for l in range(12)])
            nsel = (x >= 0).sum(-1)
            r.update(blk_set_agree_pct=100 * float(same.mean()),
                     blk_elem_agree_pct=100 * float(inter.sum() / nsel.sum()),
                     blk_set_agree_pct_margin_gt_1e3=100 * float(same[mg > 1e-3].mean()) if (mg > 1e-3).any() else None,
                     blk_set_agree_per_layer=[round(100 * float(same[l].mean()), 2) for l in range(12)],
                     blk_mismatch_kernel_margin_max=float(mg[~same].max()) if (~same).any() else None)
            pos_ok = same.all(0).all(-1)                       # positions where every NSA layer/kv-head agrees
            pm = pos_max.cpu().numpy()
            r.update(n_pos_blk_all_agree=int(pos_ok.sum()),
                     max_abs_pos_blk_all_agree=float(pm[pos_ok].max()),
                     max_abs_pos_blk_some_differ=float(pm[~pos_ok].max()) if (~pos_ok).any() else None)
        rows.append(r)
        print('compare', r, flush=True)
    info['compare'] = rows


def main():
    if os.environ.get('FAULT_SEC'):  # debugging aid: periodic Python stack dumps
        faulthandler.dump_traceback_later(int(os.environ['FAULT_SEC']), repeat=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['kernel', 'fp32', 'greedy', 'compare'])
    ap.add_argument('--prompts', default=','.join(PROMPTS))
    ap.add_argument('--kv-fake-quant', choices=['q8_0', 'q4_0'], help='fp32 stage only: fake-quantize NSA k/v')
    args = ap.parse_args()
    if args.kv_fake_quant and args.stage != 'fp32':
        ap.error('--kv-fake-quant needs --stage fp32')
    names = args.prompts.split(',')
    info_path = os.path.join(HERE, f'info_{args.stage}' + (f'_kv{args.kv_fake_quant}' if args.kv_fake_quant else '') + '.json')
    info = {}
    if args.stage == 'compare':
        stage_compare(names, info)
    else:
        model = load_model()
        if args.stage == 'greedy':
            stage_greedy(model, names, info)
        elif args.stage == 'fp32':
            stage_fp32(model, names, info, args.kv_fake_quant)
        else:
            stage_kernel(model, names, info)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=1)


if __name__ == '__main__':
    main()
