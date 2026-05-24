#!/usr/bin/env python3
"""Run agentic_prompts.json against a target endpoint and save results.

Usage:
  python3 run_agentic_bench.py <host:port> <model> <out.json>
  e.g. python3 run_agentic_bench.py http://10.117.1.215:8096 minimax-m2.7 results_minimax.json
"""
import json, sys, time, urllib.request, os

if len(sys.argv) < 4:
    print("usage: run_agentic_bench.py <base_url> <model> <out.json>")
    sys.exit(1)

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
OUT = sys.argv[3]
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_prompts.json")

# Token caps per category — give reasoning enough room
MAX_TOK = {
    "planning":  2500,
    "code":      3000,
    "math":      2500,
    "synthesis": 2500,
    "longctx":   1500,
}

def call(prompt, max_tok, timeout=900):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role":"user","content":prompt}],
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
    print(f"endpoint: {BASE}  model: {MODEL}  prompts: {len(prompts)}")
    results = []
    total_t0 = time.time()
    for i, p in enumerate(prompts, 1):
        max_tok = MAX_TOK.get(p["category"], 2000)
        print(f"  [{i:2d}/{len(prompts)}] {p['id']} {p['category']:9s}", end="", flush=True)
        r = call(p["prompt"], max_tok)
        entry = {**p, "result": r}
        results.append(entry)
        # incremental save in case of crash
        json.dump({"endpoint": BASE, "model": MODEL,
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
