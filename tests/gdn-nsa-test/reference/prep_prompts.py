# Build prompts: short + first 600/2048/4000 tokens (incl. BOS) of fineweb-edu sample-10BT.
import os, sys
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = os.environ.get('GDN_NSA_GOLDEN', 'golden')
os.makedirs(OUT, exist_ok=True)
CKPT = os.environ.get('GDN_NSA_CKPT', 'ckpt')
tok = AutoTokenizer.from_pretrained(CKPT)
assert tok.add_bos_token

ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
docs = []
for ex in ds:
    docs.append(ex['text'])
    text = '\n\n'.join(docs)
    if len(tok(text).input_ids) >= 4200:
        break
with open(os.path.join(OUT, 'text_long.txt'), 'w', encoding='utf-8') as f:
    f.write(text)
print(f'{len(docs)} docs, {len(text)} chars')

ids_long = tok(text).input_ids  # add_bos=True via tokenizer_config
assert ids_long[0] == tok.bos_token_id
prompts = {'short': tok('The capital of France is').input_ids}
for name, n in [('len600', 600), ('len2k', 2048), ('len4k', 4000)]:
    prompts[name] = ids_long[:n]
for name, ids in prompts.items():
    with open(os.path.join(OUT, f'tokens_{name}.txt'), 'w') as f:
        f.write(' '.join(map(str, ids)) + '\n')
    print(name, len(ids), ids[:8])
