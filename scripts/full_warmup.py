#!/usr/bin/env python3
"""
Warmup v3 - avoids the 32k=max_num_batched_tokens boundary that locks up the server.
Phase 3 uses 28k and 60k (instead of 32k and 64k) to force chunked prefill, avoiding the
single-chunk fast path bug at exactly max_num_batched_tokens.
"""
import sys, json, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
sys.stdout.reconfigure(line_buffering=True)

def call(msgs_or_str, max_tokens=60, timeout=1800, tools=None):
    if isinstance(msgs_or_str, str):
        msgs = [{"role": "user", "content": msgs_or_str}]
    else:
        msgs = msgs_or_str
    payload = {"model": "deepseek-v4-flash", "messages": msgs, "max_tokens": max_tokens}
    if tools: payload["tools"] = tools
    t0 = time.time()
    try:
        req = urllib.request.Request(f"{BASE}/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        elapsed = time.time() - t0
        return elapsed, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], d["choices"][0]["finish_reason"]
    except Exception as e:
        return None, 0, 0, str(e)[:120]

def wait_for_server(max_wait=600):
    print("Waiting for server...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"Ready after {time.time()-t0:.0f}s\n")
                    return True
        except Exception: pass
        time.sleep(10)
    return False

if not wait_for_server(): sys.exit(1)

# Phase 1: Prefill chunks (excluding 32768 boundary)
print("="*65)
print("Phase 1: Prefill chunk sizes (skip 32768 boundary)")
print("="*65)
chunk_sizes = [1, 16, 32, 64, 128, 192] + list(range(256, 32513, 256))  # stop at 32512
ok = fail = 0
for i, sz in enumerate(chunk_sizes):
    t, pt, ct, fin = call("x " * sz, max_tokens=4)
    if t: ok += 1
    else:
        fail += 1
        print(f"  FAIL sz={sz}: {fin}", flush=True)
    if i % 30 == 0:
        print(f"  [{i+1:3d}/{len(chunk_sizes)}] sz={sz:6d}  ok={ok}  fail={fail}", flush=True)
print(f"Phase 1: {ok}/{len(chunk_sizes)} ok\n")

# Phase 2: FreeLinus shapes
print("="*65)
print("Phase 2: FreeLinus shapes")
print("="*65)
SOUL = "You are FreeLinus, a senior software engineer at Tribble AI. Direct, technical, uses tools actively to accomplish tasks. You verify all work with actual execution and prefer minimal correct solutions."
TOOLS = [
    {"type":"function","function":{"name":"bash","description":"Execute bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read file","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"python","description":"Run python","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}
]
for label, user_msg, max_toks in [
    ("tool call",  "Check disk space.", 200),
    ("code short", "Write a Python JSON parser.", 400),
    ("long code",  "Write a complete Python async HTTP server.", 1500),
]:
    msgs = [{"role":"system","content":SOUL},{"role":"user","content":user_msg}]
    t, pt, ct, fin = call(msgs, max_tokens=max_toks, timeout=300, tools=TOOLS)
    if t: print(f"  {label:11s}  pt={pt:4d}  ct={ct:4d}  t={t:5.1f}s  {ct/t:.1f}/s  [{fin}]", flush=True)
    else: print(f"  {label:11s}  FAIL: {fin}", flush=True)
print("Phase 2 complete\n")

# Phase 3: Long context (avoid 32k boundary - use 28k and 60k to force chunked prefill)
print("="*65)
print("Phase 3: Long context (avoids 32k=max_batched boundary that locked up)")
print("="*65)
word = "the quick brown fox jumps over the lazy dog and the cat sat on the mat "
for ctx_tokens, label in [
    ( 8_000, " 8k"),
    (16_000, "16k"),
    (28_000, "28k"),  # below 32768 boundary
    (40_000, "40k"),  # forces 2 chunks
    (60_000, "60k"),  # forces 2 chunks
]:
    chars_needed = ctx_tokens * 4
    reps = (chars_needed // len(word)) + 1
    prompt = (word * reps)[:chars_needed] + "\n\nSummarize in one sentence."
    print(f"  ctx={label} ... ", end="", flush=True)
    t, pt, ct, fin = call(prompt, max_tokens=20, timeout=1800)
    if t: print(f"pt={pt:7d}  ct={ct:3d}  t={t:7.1f}s  [{fin}]", flush=True)
    else: print(f"FAIL: {fin}", flush=True)
print("Phase 3 complete\n")

# Phase 4: Decode
print("="*65)
print("Phase 4: Decode levels")
print("="*65)
for max_toks in [100, 500, 1000, 2000]:
    t, pt, ct, fin = call("Write a detailed technical essay on distributed systems.", max_tokens=max_toks, timeout=600)
    if t: print(f"  max_tokens={max_toks:5d}  ct={ct:4d}  t={t:6.1f}s  {ct/t:.1f}/s  [{fin}]", flush=True)
print("Phase 4 complete\n")

print("="*65)
print("FULL WARMUP COMPLETE")
print("="*65)
