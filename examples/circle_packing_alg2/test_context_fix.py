#!/usr/bin/env python3
"""Reproduce the exact context-overflow failure (large prompt + 12288 output tokens)
that 400-ed against the old max-model-len 16384, and confirm it now succeeds."""
from openai import OpenAI

c = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

# Prompt deliberately far larger than the old 4096-token budget.
big = open("initial.py").read() + "\n" + "x = 1  # filler line to grow the prompt\n" * 1500
msg = "Reply with just OK. Here is a program:\n\n" + big

r = c.chat.completions.create(
    model="qwen32b",
    messages=[{"role": "user", "content": msg}],
    max_tokens=12288,
    temperature=0.7,
)
u = r.usage
print(f"  prompt_tokens  = {u.prompt_tokens}   (old config left only 4096 -> would 400)")
print(f"  max_tokens     = 12288")
print(f"  total required = {u.prompt_tokens + 12288}  vs max_model_len 32768")
print(f"  completion     = {u.completion_tokens} tokens")
print(f"  reply          = {(r.choices[0].message.content or '')[:50]!r}")
assert u.prompt_tokens > 4096, "prompt was not actually large -- test invalid"
print("\n  PASS: a prompt that would have overflowed the old limit now succeeds.")
