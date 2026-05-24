#!/usr/bin/env python3
"""V2: Same 30 prompts, BUT with two interventions applied (per the judge
recommendations from the first run):

  1. System prompt nudge: put final answer first, reasoning after.
  2. Raised max_tokens per category.

Usage:
  python3 run_agentic_bench_v2.py <base_url> <model> <out.json>
"""
import json, sys, time, urllib.request, os

if len(sys.argv) < 4:
    print("usage: run_agentic_bench_v2.py <base_url> <model> <out.json>")
    sys.exit(1)

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
OUT = sys.argv[3]
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_prompts.json")

# v2 interventions
SYSTEM_PROMPT = (
    "Output your FINAL deliverable answer FIRST in a clearly-marked [FINAL] block. "
    "Put any reasoning, alternative considerations, or scratch work AFTER the [FINAL] block. "
    "Do not preamble. Do not narrate what you are about to do. Start with [FINAL] immediately."
)

# Bigger token caps to leave headroom even if reasoning expands
MAX_TOK = {
    "planning":  5000,
    "code":      6000,
    "math":      5000,
    "synthesis": 5000,
    "longctx":   3500,
}

def call(prompt, max_tok, timeout=1200):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tok,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        wall = time.time() - t0
    except Exception as e:
        return {"err": str(e), "wall": time.time()-t0}

    msg = d["choices"][0]["message"]
    u = d.get("usage", {})
    tg = d.get("timings", {})
    return {
        "wall_s": wall,
        "content": msg.get("content", ""),
        "reasoning_content": msg.get("reasoning_content", ""),
        "finish": d["choices"][0].get("finish_reason"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "decode_tps": tg.get("predicted_per_second"),
        "prefill_tps": tg.get("prompt_per_second"),
        "tpot_ms": tg.get("predicted_per_token_ms"),
    }

def main():
    prompts = json.load(open(PROMPTS_FILE))["prompts"]
    print(f"endpoint: {BASE}  model: {MODEL}  prompts: {len(prompts)}  "
          f"system_prompt=v2-answer-first  max_tokens=raised")
    results = []
    total_t0 = time.time()
    for i, p in enumerate(prompts, 1):
        max_tok = MAX_TOK.get(p["category"], 4000)
        print(f"  [{i:2d}/{len(prompts)}] {p['id']} {p['category']:9s}", end="", flush=True)
        r = call(p["prompt"], max_tok)
        results.append({**p, "result": r})
        json.dump({"endpoint": BASE, "model": MODEL,
                   "system_prompt": SYSTEM_PROMPT,
                   "max_tokens": MAX_TOK,
                   "wall_total_s": time.time() - total_t0,
                   "results": results}, open(OUT, "w"), indent=1)
        if "err" in r:
            print(f"  ERR {r['err'][:80]}")
        else:
            print(f"  out={r['completion_tokens']:5d}  "
                  f"wall={r['wall_s']:6.1f}s  "
                  f"decode={r['decode_tps'] or 0:5.2f}  "
                  f"prefill={r['prefill_tps'] or 0:6.1f}  "
                  f"finish={r['finish']}")

    print(f"\nDONE. total wall = {time.time()-total_t0:.1f}s. results -> {OUT}")

if __name__ == "__main__":
    main()
