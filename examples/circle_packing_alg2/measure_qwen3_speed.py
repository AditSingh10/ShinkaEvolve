#!/usr/bin/env python3
"""Measure real Qwen3 throughput on the ACTUAL circle-packing proposal workload,
so we can estimate wall-clock for a 200-generation run."""
import concurrent.futures as cf
import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

SEED = open("initial.py").read()
SYS = ("You are an expert mathematician specializing in circle packing and computational "
       "geometry. The best known sum of radii for 26 circles in a unit square is 2.635.")
USER = f"""# Current program

```python
{SEED}
```

Here are the performance metrics of the program:

Combined score to maximize: 0.96
num_circles: 26

# Task
Rewrite the code between the EVOLVE-BLOCK markers to improve the combined_score.
Respond with <NAME>, <DESCRIPTION> and the full rewritten code block.
"""


def one_call():
    t0 = time.time()
    r = client.chat.completions.create(
        model="qwen32b",
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": USER}],
        max_tokens=8192, temperature=0.7,
    )
    dt = time.time() - t0
    u = r.usage
    return dt, u.completion_tokens, u.prompt_tokens


print("== single request (realistic full-rewrite proposal) ==")
dt, out, inp = one_call()
print(f"  {dt:6.1f}s   output={out} tokens  prompt={inp}  -> {out/dt:.1f} tok/s")

print("\n== 4 concurrent (matches max_proposal_jobs=4) ==")
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(lambda _: one_call(), range(4)))
wall = time.time() - t0
tot_out = sum(r[1] for r in res)
print(f"  wall={wall:.1f}s  total_output={tot_out} tokens  aggregate={tot_out/wall:.1f} tok/s")
print(f"  per-request: {[f'{r[0]:.0f}s/{r[1]}tok' for r in res]}")

# ---- extrapolate to a 200-generation run
per_gen_tokens = tot_out / 4
gens = 200
eff = tot_out / wall                      # aggregate tok/s with 4-way batching
secs = gens * per_gen_tokens / eff
print(f"\n== estimate for {gens} generations ==")
print(f"  mean output/proposal : {per_gen_tokens:.0f} tokens")
print(f"  aggregate throughput : {eff:.1f} tok/s")
print(f"  LLM time             : {secs/3600:.1f} h")
print(f"  +30% eval/overhead   : {secs*1.3/3600:.1f} h   <- realistic single-run estimate")
print(f"  treatment + control  : {2*secs*1.3/3600:.1f} h")
