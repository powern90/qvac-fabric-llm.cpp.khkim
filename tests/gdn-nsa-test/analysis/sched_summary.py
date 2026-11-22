#!/usr/bin/env python3
"""Summarizes GGML_SCHED_DEBUG=2 logs: graph nodes per backend, nodes not on the GPU, and where the NSA ops ran.

usage: sched_summary.py <log> [<log> ...]
"""
import collections
import re
import sys

NODE = re.compile(r"node #\s*\d+ \(\s*(\S+)\):\s*(\S+).*?\[\s*(\S+)\s*\]")

for path in sys.argv[1:]:
    per_backend, off_gpu, nsa = collections.Counter(), collections.Counter(), collections.Counter()
    for line in open(path, errors="replace"):
        m = NODE.match(line)
        if not m:
            continue
        op, name, backend = m.groups()
        per_backend[backend] += 1
        if not backend.startswith("CUDA"):
            off_gpu[f"{op}:{name.split('-')[0]}"] += 1
        if op.startswith("NSA"):
            nsa[f"{op}@{backend}"] += 1
    print(f"{path}: nodes per backend {dict(per_backend)}; non-GPU nodes {dict(off_gpu)}; NSA nodes {dict(nsa)}")
