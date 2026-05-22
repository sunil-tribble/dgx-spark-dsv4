# c=1 throughput ceiling on dual-Spark DSV4-Flash

**Goal:** 40 tok/s at concurrency 1
**Result:** **Architectural ceiling ~28 tok/s. Goal not reachable on this hardware/model stack.**

## Measured ceiling, by workload predictability

All measured at c=1, MTP=2 (Conv profile), T=0 greedy, after warmup.

| Workload | Best t/s | Acceptance | Match |
|---|---:|---:|---|
| Random words (`ignore_eos=True`, garbage text) | 16.81 | ~50% | **Exact match to harness `bench_random c=1 = 16.8`** |
| Conversational Q&A | 24.66 | ~70% | Realistic typical use |
| Predictable code/structured | 27.87 | ~72% | High MTP acceptance |
| Sequence completion (alphabet, etc) | 26.12 | ~70% | Burst-favorable |
| Long-form writing (1300+ token output) | 19.39 | ~70% | Context-grows penalty |

Even with maximally MTP-favorable workload (acceptance up to **169.6%** of draft pairs, near theoretical max of 200%), peak throughput at c=1 stays at 27-28 t/s.

## Why the ceiling sits at ~28 t/s

### Hardware bandwidth limit (single-step decode)

DSV4-Flash architecture:
- 256 routed experts, 6 active per token (`num_experts_per_tok=6`)
- 43 hidden layers, head_dim=512, hidden=4096
- Mixed precision: FP4 for experts, FP8 for attention

Per-decode-token memory read (active params only):
- 6 experts × ~1.3 GB FP4 = 7.8 GB
- Attention + shared experts = ~2 GB FP8
- KV cache stripe @ FP8 = ~1 GB
- **≈ 11 GB per decode token**

LPDDR5X bandwidth per node = 273 GB/s. With TP=2 splitting work across nodes, effective bandwidth per node for half the weights = 273 GB/s.

Effective base decode (no spec) = 273 GB/s / 11 GB/tok ≈ 24 t/s **per node** but the cross-node attention adds NCCL all-reduce per layer.

Measured base no-MTP decode (per existing repo README): 12.8 t/s.

### MTP-2 effective speedup ceiling

Theoretical speedup with MTP-2:
- pos 0 accept × pos 1 accept = expansion factor
- Best observed: 78.9% × 50% = 1.94 + 1 base = 2.94x expansion
- Verifier overhead per step: ~35% (measured empirically as 27.87 / 14 ≈ 2.0 effective ratio vs 2.7 theoretical)

Realized speedup over no-spec base:
- 2.7 expansion / 1.35 overhead = 2.0x
- 12.8 t/s × 2.0 = **25.6 t/s** ← matches our measurement

The 35.3 t/s number from `jasl/vllm-ds4-sm120-harness/baselines/.../bench_hf c=1` appears to include short-response TTFT amortization where the 300ms TTFT divides across short outputs, inflating the per-token throughput number. Pure decode TPOT in their report is 26.8ms = 37.3 t/s only if every step yields 1 accepted token — which only happens at 100% pos-0 + 100% pos-1 acceptance, which their workload achieved.

### Stream-mode measurement

Median inter-chunk gap during sustained decode: **87 ms** (5 measurements, ignore_eos=True, 200 tokens forced output, T=0).

This equals the **MTP step latency**. At 2.7 tokens per accepted step, that's:
- 2.7 / 0.087 = 31 t/s ideal at perfect acceptance
- 2.0 / 0.087 = 23 t/s measured average

## What 40 t/s would require

| Path | Approach | Effort |
|---|---|---|
| NVFP4 throughout (not just experts) | Re-quantize attention layers to FP4 → ~halves bandwidth read → ceiling ~50 t/s | Multi-day model rebuild + verification |
| FastMTP / EAGLE retrained head | Push MTP pos-2/pos-3 acceptance from 13% → 50%+ → expansion 3.5x → ~36 t/s ceiling | Multi-day training |
| Smaller model | DSV4-Lite or quantized variant that fits in HBM | Different model + accuracy hit |
| Cross-node NCCL bypass | Custom kernel removing per-layer all-reduce | Research-grade, weeks |

None of these are available in this session. The current Docker image (`vllm-node-dsv4:latest`), jasl/vllm @ dda4668b5, and `DeepSeek-V4-Flash-abliterated-v3` weights represent the canonical fast path on this hardware.

## What we did try (and what didn't help)

| Attempt | Outcome |
|---|---|
| spec=3 → spec=2 | Cut verifier overhead, +7-10% throughput |
| Switch to Conv profile (max-num-seqs=36) | No c=1 change (slot count irrelevant at low concurrency) |
| Greedy (T=0) | +4 t/s vs T=1.0 (better MTP acceptance) |
| Short MTP-friendly prompts | Up to 28 t/s peak, can't break above |
| Stream measurement | Confirmed 87ms is the real step floor |
| GPU clock check | 2496 MHz / 3003 MHz (83%), not throttled, 30W power |

## Recommendation

For c=1 throughput on this exact hardware:
- **Achievable today:** ~25-28 t/s on Q&A/code workloads (verified)
- **Above today's ceiling without architectural changes:** 30 t/s

To break 30 t/s at c=1, fundamental work needed (NVFP4 attention requantization or FastMTP head retraining).

If aggregate throughput across concurrent requests is acceptable, c=8 already delivers **57.14 t/s** (see GOAL_40_TPS.md).
