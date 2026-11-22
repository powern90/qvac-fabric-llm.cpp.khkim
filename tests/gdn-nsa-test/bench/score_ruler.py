#!/usr/bin/env python
"""Score the RULER S-NIAH subset for FLA, ggml CPU and ggml GPU with RULER's niah metric.

RULER scripts/eval/synthetic/constants.py: niah -> string_match_all
    score = mean over samples of (fraction of reference strings r with r.lower() in pred.lower()) * 100
predictions: HF tokenizer decode(skip_special_tokens=True) + RULER evaluate.postprocess_pred (strip, control chars
-> '\\n', strip).
Writes ruler_report.md and ruler.json. A backend with fewer samples (truncated CPU run) is scored on the samples it
has, and its delta / agreement are computed against FLA on the same samples.
"""
import argparse, glob, json, os, re
from collections import OrderedDict

PAPER = {  # laNSA 0.4B, Table 2
    ('niah_single_1', 1024): 100.0, ('niah_single_1', 2048): 100.0, ('niah_single_1', 4096): 65.0,
    ('niah_single_2', 1024): 88.6, ('niah_single_2', 2048): 100.0, ('niah_single_2', 4096): 37.6,
    ('niah_single_3', 1024): 97.6, ('niah_single_3', 2048): 73.0, ('niah_single_3', 4096): 11.8,
}
PAPER_AVG = 74.8
SHORT = {'niah_single_1': 'S1', 'niah_single_2': 'S2', 'niah_single_3': 'S3'}
NP = re.compile(r'[\x00-\x1f]')


def postprocess_pred(s):
    s = s.strip()
    return NP.sub('\n', s).strip()


def string_match_all_one(pred, ref):
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)


def load_gen(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if p:
            out[p[0]] = [int(x) for x in p[1:]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='ruler')
    ap.add_argument('--ckpt', default='ckpt')
    ap.add_argument('--max-new', type=int, default=64, help='generation length of the runs (for the label)')
    ap.add_argument('--trunc', type=int, default=0, help='score only the first N generated tokens (greedy prefix = '
                    'an N-token run); outputs get the suffix _N')
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    samples = [json.loads(l) for l in open(os.path.join(args.dir, 'ruler_niah.jsonl'))]
    backends = OrderedDict(fla='gen_fla.txt', cpu='gen_ggml_cpu.txt', gpu='gen_ggml_cuda.txt')
    gens = {b: load_gen(os.path.join(args.dir, f)) for b, f in backends.items()}
    # FLA may run as several workers on one GPU (gen_fla_*.txt); merge them and check that overlaps agree
    fla_dup = fla_dup_diff = 0
    for extra in sorted(glob.glob(os.path.join(args.dir, 'gen_fla_*.txt'))):
        for i, g in load_gen(extra).items():
            if i in gens['fla']:
                fla_dup += 1
                fla_dup_diff += gens['fla'][i] != g
            else:
                gens['fla'][i] = g
    per = {}  # backend -> id -> dict
    for b in backends:
        per[b] = {}
        for s in samples:
            if s['id'] not in gens[b]:
                continue
            g = gens[b][s['id']][:args.trunc] if args.trunc else gens[b][s['id']]
            pred = postprocess_pred(tok.decode(g, skip_special_tokens=True))
            per[b][s['id']] = dict(gen=g, pred=pred, score=string_match_all_one(pred, s['answer']))

    cells = OrderedDict()
    for s in samples:
        cells.setdefault((s['task'], s['length']), []).append(s)
    cells = OrderedDict(sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])))

    def acc(b, ids):
        ids = [i for i in ids if i in per[b]]
        return (100 * sum(per[b][i]['score'] for i in ids) / len(ids), len(ids)) if ids else (None, 0)

    rows, js = [], dict(cells={}, disagreements={}, label='RULER S-NIAH subset (our generator with RULER templates)')
    tot = {b: [] for b in backends}
    for (task, L), ss in cells.items():
        ids = [s['id'] for s in ss]
        r = dict(task=task, length=L, n=len(ids), paper=PAPER[(task, L)])
        for b in backends:
            r[b], r[f'n_{b}'] = acc(b, ids)
        for b in ('cpu', 'gpu'):
            common = [i for i in ids if i in per[b] and i in per['fla']]
            r[f'n_common_{b}'] = len(common)
            if common:
                fa = 100 * sum(per['fla'][i]['score'] for i in common) / len(common)
                ga = 100 * sum(per[b][i]['score'] for i in common) / len(common)
                r[f'delta_{b}'] = ga - fa
                r[f'tok_agree_{b}'] = sum(per[b][i]['gen'] == per['fla'][i]['gen'] for i in common)
                r[f'corr_agree_{b}'] = sum(per[b][i]['score'] == per['fla'][i]['score'] for i in common)
                dis = []
                for i in common:
                    a, g = per['fla'][i]['gen'], per[b][i]['gen']
                    if a != g:
                        fd = next((k for k, (x, y) in enumerate(zip(a, g)) if x != y), min(len(a), len(g)))
                        dis.append(dict(id=i, first_diff_step=fd, fla_score=per['fla'][i]['score'], score=per[b][i]['score'],
                                        fla_pred=per['fla'][i]['pred'], pred=per[b][i]['pred']))
                js['disagreements'].setdefault(b, []).extend(dis)
            else:
                r[f'delta_{b}'] = r[f'tok_agree_{b}'] = r[f'corr_agree_{b}'] = None
        for b in backends:
            tot[b].extend(i for i in ids if i in per[b])
        rows.append(r)
        js['cells'][f'{SHORT[task]}_{L}'] = r

    f1 = lambda x: '-' if x is None else f'{x:.1f}'
    fd = lambda x: '-' if x is None else f'{x:+.1f}'
    def agree(r, b):
        if r[f'tok_agree_{b}'] is None:
            return '-'
        n = r[f'n_common_{b}']
        return f"{r[f'tok_agree_{b}']}/{n} tok, {r[f'corr_agree_{b}']}/{n} corr"
    n_per = max(r['n'] for r in rows)
    n_gen = args.trunc or args.max_new
    lines = [f'# RULER S-NIAH subset (N = {n_per} per task and length, our generator with RULER templates)', '',
             'Not a reproduction of the paper: own sample generator (RULER niah.py logic and RULER strings), '
             f'greedy decoding with max_new_tokens = {n_gen} (RULER: 128), RULER string_match_all scoring. '
             'Paper = laNSA 0.4B, Table 2. FLA = HF/fla reference (author kernel path, fp32 weights) on GPU; '
             'ggml = gdn-nsa-0.4B-f32.gguf, F16 KV cache. Delta = ggml - FLA in pp on the same samples. '
             'agree = identical generated token sequence (tok) / same per-sample correctness (corr).', '']
    ncpu = sum(r['n_cpu'] for r in rows)
    if ncpu < sum(r['n'] for r in rows):
        lines.append(f'ggml CPU covers {ncpu} of {sum(r["n"] for r in rows)} samples (first K per task/length, '
                     f'K = {min(r["n_cpu"] for r in rows)}..{max(r["n_cpu"] for r in rows)}); its columns use those samples only.')
        lines.append('')
    lines += ['| Task | Paper | FLA | ggml CPU | ggml GPU | Δ CPU | Δ GPU | agree CPU | agree GPU |',
              '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        name = f"{SHORT[r['task']]} {r['length'] // 1024}K"
        cpu_s = f1(r['cpu']) + (f" (n={r['n_cpu']})" if r['n_cpu'] != r['n'] else '')
        lines.append(f"| {name} | {r['paper']:.1f} | {f1(r['fla'])} | {cpu_s} | {f1(r['gpu'])} | {fd(r['delta_cpu'])} | "
                     f"{fd(r['delta_gpu'])} | {agree(r, 'cpu')} | {agree(r, 'gpu')} |")
    # averages: mean of the cell accuracies (the paper's average is the mean of its 9 cells)
    avg = {b: (sum(r[b] for r in rows if r[b] is not None) / len([r for r in rows if r[b] is not None])
               if any(r[b] is not None for r in rows) else None) for b in backends}
    def sumagree(b, k):
        return sum(r[f'{k}_agree_{b}'] or 0 for r in rows), sum(r[f'n_common_{b}'] for r in rows)
    ag = {b: '%d/%d tok, %d/%d corr' % (*sumagree(b, 'tok'), *sumagree(b, 'corr')) for b in ('cpu', 'gpu')}
    def mean_delta(b):
        v = [r[f'delta_{b}'] for r in rows if r[f'delta_{b}'] is not None]
        return sum(v) / len(v) if v else None
    dcpu, dgpu = mean_delta('cpu'), mean_delta('gpu')
    lines.append(f"| **Average** | {PAPER_AVG:.1f} | {f1(avg['fla'])} | {f1(avg['cpu'])} | {f1(avg['gpu'])} | {fd(dcpu)} | "
                 f"{fd(dgpu)} | {ag['cpu']} | {ag['gpu']} |")
    lines.append('')
    for b in ('cpu', 'gpu'):
        dis = js['disagreements'].get(b, [])
        lines.append(f'Token-sequence disagreements FLA vs ggml {b.upper()}: {len(dis)}' + (':' if dis else ''))
        for d in dis:
            lines.append(f"- {d['id']}: first differing step {d['first_diff_step']}, score FLA {d['fla_score']:.0f} / "
                         f"ggml {d['score']:.0f}; FLA {d['fla_pred'][:60]!r} vs ggml {d['pred'][:60]!r}")
        lines.append('')
    lines.append(f'FLA worker overlap: {fla_dup} samples generated twice, {fla_dup_diff} differ.')
    js['fla_worker_overlap'] = dict(n=fla_dup, n_differ=fla_dup_diff)
    js['average'] = dict(paper=PAPER_AVG, **avg, delta_cpu=dcpu, delta_gpu=dgpu)
    js['per_sample'] = {b: {i: dict(score=v['score'], pred=v['pred']) for i, v in per[b].items()} for b in backends}
    js['answers'] = {s['id']: s['answer'] for s in samples}
    sfx = f'_{args.trunc}' if args.trunc else ''
    js['max_new_tokens'] = n_gen
    open(os.path.join(args.dir, f'ruler_report{sfx}.md'), 'w').write('\n'.join(lines) + '\n')
    json.dump(js, open(os.path.join(args.dir, f'ruler{sfx}.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
