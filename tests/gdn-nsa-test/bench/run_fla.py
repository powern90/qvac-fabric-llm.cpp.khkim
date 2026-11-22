#!/usr/bin/env python
"""FLA reference generation for the RULER S-NIAH subset (author's kernel path).

Model loading is tests/gdn-nsa-test/reference/run_ref.py:load_model (fp32 weights, fla kernels, fp16 q/k/v casts inside
the layers, flash-attn sliding window, TF32 off), the same as its "greedy" stage. Per sample:
generate(do_sample=False, max_new_tokens=N, eos_token_id=2, use_cache=True).
Output: one line per sample "id gen1 gen2 ..." (EOS included when produced), same format as nsa-generate.

  python run_fla.py --ckpt ckpt --data ruler_niah.jsonl --out gen_fla.txt
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='ckpt')
ap.add_argument('--data', default='ruler/ruler_niah.jsonl')
ap.add_argument('--out', default='ruler/gen_fla.txt')
ap.add_argument('--max-new', type=int, default=64)
ap.add_argument('--start', type=int, default=0, help='first sample index (for parallel workers on one GPU)')
ap.add_argument('--reverse', action='store_true', help='walk the samples backwards from --start (default: last)')
args = ap.parse_args()

os.environ['GDN_NSA_CKPT'] = args.ckpt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference'))
sys.argv = sys.argv[:1]
import torch  # noqa: E402
from run_ref import load_model  # noqa: E402


@torch.no_grad()
def main():
    model = load_model()
    samples = [json.loads(l) for l in open(args.data)]
    samples = samples[args.start:] if not args.reverse else samples[:(args.start or len(samples))][::-1]
    t0 = time.time()
    with open(args.out, 'w') as f:
        for k, s in enumerate(samples):
            ids = torch.tensor([s['input_ids']], device='cuda')
            out = model.generate(ids, max_new_tokens=args.max_new, min_new_tokens=0, do_sample=False, use_cache=True,
                                 eos_token_id=2, pad_token_id=0)
            gen = out[0, ids.shape[1]:].tolist()
            f.write(s['id'] + ' ' + ' '.join(map(str, gen)) + '\n')
            f.flush()
            print(f'[{k + 1}/{len(samples)}] {s["id"]} T={ids.shape[1]} gen={len(gen)} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
