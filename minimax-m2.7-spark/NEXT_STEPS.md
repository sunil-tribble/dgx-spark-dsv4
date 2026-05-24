# MiniMax-M2.7 on Spark — paths to push past 24 t/s

Current locked-in baseline: **23.55 t/s mean / 23.92 peak at 192K context**, ~86% of LPDDR5X bandwidth roofline. Confirmed bandwidth-bound: more threads (12→16) and bigger batch (b=512→1024) made zero difference.

To go higher requires changing the constraint. Three viable paths:

## A — Dual-Spark RPC (highest upside, biggest effort)

Split MiniMax-M2.7 weights across both Sparks via llama.cpp's RPC backend.
2× aggregate LPDDR5X bandwidth, ~30-50% realized gain after serialization
overhead.

**Expected**: ~30-35 t/s decode, possibly higher at long ctx
**Effort**: 4-8h
**Steps**:
1. Rebuild llama.cpp on both Sparks with `-DGGML_RPC=ON`
2. Start `rpc-server` on spark-1 listening on the RoCE network
3. Launch llama-server on spark-2 with `--rpc 192.168.100.1:50052 -ts 1,1`
4. Bench

Current build has `llama_supports_rpc` symbol but not the CLI flag — confirmed
needs rebuild.

## B — Speculative decoding (medium effort, medium upside)

Run a small draft model alongside MiniMax-M2.7. If accept rate ≥ 60% and
draft is <1B params:

**Expected**: 1.5-2× decode (35-50 t/s)
**Effort**: 1-3h, gated on finding a draft
**Blocker**: MiniMax-M2.7 uses a 200K-token custom vocab. Off-the-shelf
small models (Qwen, Llama, Mistral) have incompatible tokenizers — no useful
draft exists publicly. Would need to distill or train one (days).

Realistic next move here: wait for MiniMax to release an M2.7-Lite, or wait
for the community to ship a matching draft.

## C — Turboquant build for aarch64 (low upside, medium effort)

Rebuild `llama-cpp-turboquant` on Spark to enable `turbo4` V-cache. Saves
~3 GB at 192K (frees memory but won't speed up decode — we're not memory-
bound at active layer level).

**Expected**: same speed, frees 3 GB for larger batch/ctx headroom
**Effort**: 2-4h (CUDA aarch64 build + verify on GB10 sm_121)
**Won't help** decode t/s. Useful only if we want even longer ctx than 192K
(YaRN extrapolation to 256K+).

## Recommendation

**A** is the only path that meaningfully changes decode t/s. If aiming for
~35 t/s on MiniMax-M2.7 single-stream, schedule a dedicated rebuild session.

For now, **single-Spark 192K @ 23.5 t/s is the committed, working baseline**.
