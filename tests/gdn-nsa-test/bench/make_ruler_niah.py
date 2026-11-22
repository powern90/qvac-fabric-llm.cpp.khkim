#!/usr/bin/env python
"""RULER S-NIAH subset (niah_single_1/2/3) built with the real RULER strings.

Strings are fetched from NVIDIA/RULER (main): scripts/data/synthetic/constants.py (template + answer_prefix,
parsed with ast.literal_eval), scripts/data/synthetic/niah.py (needle, noise haystack, parsed by regex) and
scripts/synthetic.yaml (task configs, checked). Haystack essays: gkamradt/LLMTest_NeedleInAHaystack
needlehaystack/PaulGrahamEssays/*.txt (listed through the GitHub API), sorted by file name, joined with " ".

The sample construction follows RULER niah.py generate_input_output/generate_samples (binary search on the
haystack size so that len(tokenize(input)) + tokens_to_generate <= max_seq_length, tokens counted without BOS as
RULER's hf tokenizer wrapper does; random.seed(42) per (task, length) like separate RULER invocations).
Deviations: when a sample overflows, the haystack shrinks by 10 essay words / 1 noise sentence instead of RULER's
500 / 25 (RULER's step drops e.g. a 1K essay sample to ~230 tokens); the needle depth of sample i is fixed at i / (N - 1) * 100 % (uniform spread) instead of random.sample(DEPTHS).
The prompt is input + answer_prefix (what RULER / lm-eval feed a base model), tokenized with the HF tokenizer
(BOS added by the tokenizer).

needs nltk (punkt) and wonderwords: PYTHONPATH=<dir with them> NLTK_DATA=<punkt dir>
"""
import argparse, ast, glob, json, os, random, re, urllib.request, uuid

import numpy as np

RULER = 'https://raw.githubusercontent.com/NVIDIA/RULER/main/'
PG_API = 'https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/needlehaystack/PaulGrahamEssays'
TASKS = {  # scripts/synthetic.yaml (checked against the fetched file below)
    'niah_single_1': dict(type_haystack='noise', type_needle_k='words', type_needle_v='numbers'),
    'niah_single_2': dict(type_haystack='essay', type_needle_k='words', type_needle_v='numbers'),
    'niah_single_3': dict(type_haystack='essay', type_needle_k='words', type_needle_v='uuids'),
}
SHORT = {'niah_single_1': 'S1', 'niah_single_2': 'S2', 'niah_single_3': 'S3'}


def fetch(url, cache):
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r, open(cache, 'wb') as f:
            f.write(r.read())
    return open(cache, encoding='utf-8').read()


def load_ruler_strings(cache_dir):
    const_src = fetch(RULER + 'scripts/data/synthetic/constants.py', os.path.join(cache_dir, 'ruler_constants.py.txt'))
    niah_src = fetch(RULER + 'scripts/data/synthetic/niah.py', os.path.join(cache_dir, 'ruler_niah.py.txt'))
    yaml_src = fetch(RULER + 'scripts/synthetic.yaml', os.path.join(cache_dir, 'ruler_synthetic.yaml.txt'))
    tasks = None
    for node in ast.parse(const_src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'TASKS' for t in node.targets):
            tasks = ast.literal_eval(node.value)
    niah = tasks['niah']
    needle = re.search(r'^needle = "(.*)"$', niah_src, re.M).group(1)
    noise = re.search(r"type_haystack == 'noise':\n\s+haystack = \"(.*)\"", niah_src).group(1)
    for name, cfg in TASKS.items():  # check the task configs against synthetic.yaml
        blk = re.search(name + r':\n  task: niah\n  args:\n((?:    .*\n)+)', yaml_src).group(1)
        got = dict(re.findall(r'    (\w+): (\S+)', blk))
        for k, v in cfg.items():
            assert got[k] == v, (name, k, got[k], v)
        assert got['num_needle_k'] == got['num_needle_v'] == got['num_needle_q'] == '1', got
    return niah['template'], niah['answer_prefix'], niah['tokens_to_generate'], needle, noise


def load_essays(cache_dir):
    d = os.path.join(cache_dir, 'PaulGrahamEssays')
    if not glob.glob(os.path.join(d, '*.txt')):
        os.makedirs(d, exist_ok=True)
        with urllib.request.urlopen(PG_API, timeout=30) as r:
            listing = json.loads(r.read())
        for it in listing:
            if it['name'].endswith('.txt'):
                fetch(it['download_url'], os.path.join(d, it['name']))
    files = sorted(glob.glob(os.path.join(d, '*.txt')))
    return ' '.join(open(f, encoding='utf-8').read() for f in files), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpt')
    ap.add_argument('--out', default='ruler')
    ap.add_argument('--cache', default='ruler/src')
    ap.add_argument('--lengths', default='1024,2048,4096')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-new', type=int, default=64, help='generation length written for the C++ runner')
    args = ap.parse_args()

    from nltk.tokenize import sent_tokenize
    import wonderwords
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    text_to_tokens = lambda t: tok.tokenize(t)   # RULER HFTokenizer.text_to_tokens (no BOS)

    template_t, answer_prefix, tokens_to_generate, needle, noise = load_ruler_strings(args.cache)
    essay_text, n_files = load_essays(args.cache)
    essay_words = re.sub(r'\s+', ' ', essay_text).split(' ')
    template_full = template_t + answer_prefix
    print('template:', repr(template_full))
    print('needle:', repr(needle), '\nnoise:', repr(noise))
    print(f'essays: {n_files} files, {len(essay_words)} words; tokens_to_generate (length budget) = {tokens_to_generate}')

    nouns = wonderwords.random_word._get_words_from_text_file('nounlist.txt')
    adjs = wonderwords.random_word._get_words_from_text_file('adjectivelist.txt')
    words = sorted(set(f'{a}-{n}' for a in adjs for n in nouns))
    DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

    def gen(kind):
        if kind == 'numbers':
            return str(random.randint(10 ** 6, 10 ** 7 - 1))
        if kind == 'words':
            return random.choice(words)
        if kind == 'uuids':
            return str(uuid.UUID(int=random.getrandbits(128), version=4))
        raise ValueError(kind)

    def generate_input_output(cfg, num_haystack, depth):
        """RULER niah.generate_input_output for num_needle_k = v = q = 1; depth (percent) replaces random.sample(DEPTHS)."""
        key = gen(cfg['type_needle_k'])
        value = gen(cfg['type_needle_v'])
        needles = [needle.format(type_needle_v=cfg['type_needle_v'], key=key, value=value)]
        if cfg['type_haystack'] == 'essay':
            hs = essay_words
            if num_haystack <= len(hs):
                text = ' '.join(hs[:num_haystack])
            else:
                repeats = (num_haystack + len(hs) - 1) // len(hs)
                text = ' '.join((hs * repeats)[:num_haystack])
            sents = sent_tokenize(text.strip())
            random.sample(DEPTHS, 1)  # keep RULER's RNG stream
            pos = [0, int(len(sents) * (depth / 100)), len(sents)]
            parts = []
            for i in range(1, len(pos)):
                parts.append(' '.join(sents[pos[i - 1]:pos[i]]))
                if i - 1 < len(needles):
                    parts.append(needles[i - 1])
            context = ' '.join(parts)
        else:
            sentences = [noise] * num_haystack
            random.sample(range(num_haystack), 1)  # keep RULER's RNG stream
            sentences.insert(int(round(num_haystack * depth / 100)), needles[0])
            context = '\n'.join(sentences)
        template = template_full.replace('Some', 'A').replace('are all', 'is').replace('are', 'is').replace('answers', 'answer')
        input_text = template.format(type_needle_v=cfg['type_needle_v'][:-1], context=context, query=key)
        return input_text, [value]

    os.makedirs(args.out, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(',')]
    samples = {}
    for task, cfg in TASKS.items():
        for L in lengths:
            random.seed(args.seed)
            np.random.seed(args.seed)
            incremental = 500 if cfg['type_haystack'] == 'essay' else 25
            if cfg['type_haystack'] != 'essay' and L < 4096:
                incremental = 5
            s_in, _ = generate_input_output(cfg, incremental, 50)
            tph = len(text_to_tokens(s_in)) / incremental
            lo, hi = incremental, max(int((L / tph) * 3), incremental * 2)
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                t, _ = generate_input_output(cfg, mid, 50)
                if len(text_to_tokens(t)) + tokens_to_generate <= L:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            num_haystack = best if best is not None else incremental
            step = 10 if cfg['type_haystack'] == 'essay' else 1  # deviation: finer retry step keeps samples near L
            for i in range(args.n):
                depth = 100.0 * i / max(1, args.n - 1)
                used = num_haystack
                while True:
                    t, ans = generate_input_output(cfg, used, depth)
                    length = len(text_to_tokens(t)) + tokens_to_generate
                    if length <= L:
                        break
                    used -= step  # RULER steps back by `incremental` (500 essay words / 5-25 noise sentences)
                ids = tok(t)['input_ids']
                assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
                sid = f'{SHORT[task]}_{L}_{i:02d}'
                samples[(task, L, i)] = dict(id=sid, task=task, length=L, index=i, depth=round(depth, 2),
                                             num_haystack=used, n_input=len(ids), input_ids=ids, answer=ans)
            print(task, L, 'haystack', num_haystack, 'n_input', [samples[(task, L, i)]['n_input'] for i in range(args.n)][:5], '...', flush=True)

    # round-robin order (index outer): a truncated run covers the first K samples of every (task, length)
    order = [samples[(t, L, i)] for i in range(args.n) for t in TASKS for L in lengths]
    with open(os.path.join(args.out, 'ruler_niah.jsonl'), 'w') as f:
        for s in order:
            f.write(json.dumps(s) + '\n')
    with open(os.path.join(args.out, 'ruler_niah.txt'), 'w') as f:
        for s in order:
            f.write(f"{s['id']} {args.max_new} " + ' '.join(map(str, s['input_ids'])) + '\n')
    print('wrote', len(order), 'samples; max n_input', max(s['n_input'] for s in order))


if __name__ == '__main__':
    main()
