#!/usr/bin/env python3
"""
Patch vLLM multiproc_executor.py inside container.
Fixes:
1. Follower collective_rpc guard (prevents init assertion failure)
2. check_health timeout 10s -> 3600s (THE root cause: when Triton JIT for
   batch=N x large KV blocks takes >10s, health check fails, worker is
   killed mid-NCCL all-reduce, both nodes lock up).
"""
import subprocess, sys, re

result = subprocess.run(
    ["find", "/usr/local/lib", "-name", "multiproc_executor.py", "-path", "*/vllm/*"],
    capture_output=True, text=True
)
files = [f for f in result.stdout.strip().split('\n') if f]
if not files:
    print("ERROR: multiproc_executor.py not found"); sys.exit(1)

target = files[0]
print(f"Patching: {target}")
with open(target) as f:
    content = f.read()

applied = []

# Patch 1: follower collective_rpc guard
OLD1 = "assert self.rpc_broadcast_mq is not None"
GUARD = "if self.rpc_broadcast_mq is None: return []  # follower: skip"
if GUARD in content:
    print("  Patch 1 (follower guard): already applied")
    applied.append("follower_guard_idempotent")
elif OLD1 in content:
    content = content.replace(OLD1, f"{GUARD}\n        {OLD1}", 1)
    print("  Patch 1 (follower guard): applied")
    applied.append("follower_guard")

# Patch 2: check_health timeout 10 -> 3600
OLD2 = 'self.collective_rpc("check_health", timeout=10)'
NEW2 = 'self.collective_rpc("check_health", timeout=3600)  # patched: was 10s, JIT can block longer'
if NEW2 in content:
    print("  Patch 2 (check_health timeout): already applied")
    applied.append("check_health_idempotent")
elif OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    print("  Patch 2 (check_health timeout): 10s -> 3600s applied")
    applied.append("check_health")
else:
    # Look for any "check_health" with a timeout
    m = re.search(r'collective_rpc\("check_health"[^)]*timeout=\d+\)', content)
    if m:
        old = m.group(0)
        new = re.sub(r'timeout=\d+', 'timeout=3600', old)
        content = content.replace(old, new)
        print(f"  Patch 2 (check_health regex): {old} -> {new}")
        applied.append("check_health_regex")
    else:
        print("  Patch 2 (check_health): not found")

# Patch 3: Search for any other hardcoded short timeouts in this file
short_timeouts = re.findall(r'timeout=(\d+)', content)
print(f"  Remaining timeout=N values: {short_timeouts}")

if applied:
    with open(target, 'w') as f:
        f.write(content)
    print(f"\nPatches applied: {applied}")
    print(f"File: {target}")
else:
    print("\nNo patches applied")
