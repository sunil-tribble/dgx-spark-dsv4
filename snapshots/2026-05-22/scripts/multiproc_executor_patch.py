#!/usr/bin/env python3
"""Patch multiproc_executor.py: idempotent.
1. Replace 'assert self.rpc_broadcast_mq is not None' with graceful return for followers
2. check_health timeout 10 -> 3600
"""
import pathlib
P = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py')
src = P.read_text()
orig = src

# Patch 1: replace the assert with a graceful return for followers
OLD1 = '''        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )'''
NEW1 = '''        if self.rpc_broadcast_mq is None:  # PATCH_RPC_GUARD
            return []  # follower: no work to broadcast'''
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    print('patch1 applied (rpc_broadcast_mq guard)', flush=True)
elif 'PATCH_RPC_GUARD' in src:
    print('patch1 already applied', flush=True)
else:
    print('patch1 PATTERN NOT FOUND', flush=True)

# Patch 2: check_health timeout 10 -> 3600
OLD2 = 'self.collective_rpc("check_health", timeout=10)'
NEW2 = 'self.collective_rpc("check_health", timeout=3600)  # PATCH_CHECK_HEALTH'
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    print('patch2 applied (check_health timeout)', flush=True)
elif 'PATCH_CHECK_HEALTH' in src:
    print('patch2 already applied', flush=True)
else:
    print('patch2 PATTERN NOT FOUND', flush=True)

if src != orig:
    P.write_text(src)
    print('file written', flush=True)
else:
    print('no changes', flush=True)
