#!/usr/bin/env bash
# Smoke test + quick perf bench against MiniMax-M2.7 on Spark.
# Usage: bash smoke_bench.sh [host:port]

set -euo pipefail
HOST="${1:-http://127.0.0.1:8096}"

echo "=== /health ==="
curl -fsS -m 5 "$HOST/health" || { echo "DOWN"; exit 1; }
echo

echo "=== /v1/models ==="
curl -fsS -m 5 "$HOST/v1/models" | python3 -m json.tool | head -20
echo

echo "=== smoke (greedy short) ==="
curl -fsS -m 60 "$HOST/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m2.7","messages":[{"role":"user","content":"Reply with only OK."}],"max_tokens":10,"temperature":0}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("content:", repr(d["choices"][0]["message"]["content"])); print("usage:", d.get("usage"))'
echo

echo "=== quick bench: 5 prompts, 200-tok cap, T=0 ==="
python3 - <<'PY'
import json, time, urllib.request, statistics, sys
HOST = "${HOST}".replace("$","") or "http://127.0.0.1:8096"
PROMPTS = [
    "Write a short Python function to compute the nth Fibonacci number.",
    "Explain Bayes' theorem in 3 sentences.",
    "List 5 differences between Rust and Go.",
    "Summarize the plot of 'The Odyssey' in 4 sentences.",
    "Suggest 4 names for an AI inference startup.",
]
samples=[]
for i,p in enumerate(PROMPTS,1):
    body = json.dumps({"model":"minimax-m2.7","messages":[{"role":"user","content":p}],
                       "max_tokens":300,"temperature":0,"ignore_eos":True}).encode()
    req = urllib.request.Request(HOST+"/v1/chat/completions", data=body,
                                 headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=300) as r: d=json.loads(r.read())
    dt=time.time()-t0
    ct=d.get("usage",{}).get("completion_tokens",0)
    pt=d.get("usage",{}).get("prompt_tokens",0)
    tps=ct/dt
    samples.append(tps)
    print(f"  {i}: prompt={pt} out={ct} time={dt:.2f}s = {tps:.2f} t/s")
print()
print(f"mean   = {statistics.mean(samples):.2f} t/s")
print(f"median = {statistics.median(samples):.2f} t/s")
print(f"peak   = {max(samples):.2f} t/s")
PY
