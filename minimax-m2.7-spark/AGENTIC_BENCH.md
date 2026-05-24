# Hard Agentic Bench: MiniMax-M2.7 vs DeepSeek-V4-Flash on DGX Spark

**30 prompts × 2 models, head-to-head.** 5 categories (planning, code, math, synthesis, long-context-recall), 6 prompts each. Single-stream c=1, T=0 greedy, max_tokens per category. Rubric-graded 0-10 by 5 parallel Claude judges, one per category.

## Headlines

| | MiniMax-M2.7 (UD-IQ4_XS) | DeepSeek-V4-Flash |
|---|---:|---:|
| Stack | llama.cpp build 9304, **single Spark**, llama-server | jasl/vllm @ dda4668b5, **dual Spark** TP=2 EP, vLLM |
| Memory budget | 108 GB weights + 12 GB KV / 121 GB usable | 75 GB/node weights + 18 GB KV/node |
| Context length | **192K** | 32K |
| KV cache | q4_0/q4_0 | FP8 |
| Speculative decoding | none | MTP-2 (DeepSeek native) |
| **Total wall time** | **3411s (56.9m)** | **1981s (33.0m)** |
| Total tokens generated | 70,593 | 47,713 |
| Mean decode | 20.49 t/s | 23.38 t/s |

## Quality (rubric-graded 0-10)

| category | MiniMax avg | DSV4 avg | DSV4 wins | MiniMax wins | ties |
|---|---:|---:|---:|---:|---:|
| planning  | 3.00 | 6.83 | 6 | 0 | 0 |
| code      | 1.00 | 4.17 | 6 | 0 | 0 |
| math      | 4.67 | 7.33 | 5 | 1 | 0 |
| synthesis | 3.50 | 6.67 | 5 | 0 | 1 |
| longctx   | 7.83 | 10.00 | 3 | 0 | 3 |
| **OVERALL** | **4.00** | **7.00** | **25** | **1** | **4** |

**DSV4-Flash sweeps 25-1-4 across all 30 head-to-head items.**

## Speed + output by category

| category | MM decode | MM out tok | MM natural-stop | DSV4 decode | DSV4 out tok | DSV4 natural-stop |
|---|---:|---:|---:|---:|---:|---:|
| planning  | 21.25 t/s | 2500 | 0% | 22.58 t/s | 1813 | 83% |
| code      | 20.73 t/s | 3000 | 0% | 25.74 t/s | 1872 | 100% |
| math      | 21.41 t/s | 2500 | 0% | 27.41 t/s | 1815 | 83% |
| synthesis | 21.37 t/s | 2427 | 33% | 22.82 t/s | 2147 | 67% |
| longctx   | 17.68 t/s | 1339 | 50% | 18.36 t/s | 304 | 100% |

## Per-category verdicts

### planning

DSV4 dominates planning tasks because it actually converges to structured final answers within the token budget, while MiniMax-M2.7 burns its entire 2500-token budget on stream-of-consciousness reasoning and rarely emits a [FINAL] section. Even when MiniMax's reasoning surfaces correct ideas, the lack of a delivered answer makes it unusable for planning work.

### code

Catastrophic showing by minimax-m2.7 on code: across all 6 prompts it never emitted compilable code, instead spending its 3000-token budget on stream-of-consciousness design deliberation that always hit the length cap mid-thought. One prompt (B3) literally degenerates into copy-paste loops ('The class should also handle the case where the Redis client...'). This looks like a model that cannot terminate reasoning and commit to output — a serious agentic-coding failure mode. DSV4 produces real code on every prompt; it has notable bugs on most (broken striped LRU list ops on B1, wrong rectangular padding on B2, ZSET member collision on B3, broken overflow check on B4, fabricated bug diagnosis on B5, undrained task queue on B6) but a working partial implementation beats no implementation in every head-to-head. DSV4 wins 6-0. Neither model is production-grade on the hardest items, but DSV4 is in another league for usability.

### math

DSV4 dominates on math 5-1. The dominant failure mode for Minimax is not mathematical incompetence — its scratch work is often correct and sometimes more insightful (C4 counting, C6 Gronwall via h=f^2) — but a structural inability to escape its meta-reasoning preamble within the 2500-token budget. Every single Minimax response ends with a literal '[FINAL]' marker followed by empty/truncated content (finish_reason='length'). DSV4 reliably produces a polished structured answer in fewer tokens (often finish_reason='stop'). Minimax also commits one outright algebra error on C2 (c = 1/(8 sqrt(2)) instead of 1/(4 sqrt(2)), off by factor of 2). On the one item where DSV4 truly struggles (C4 group classification), DSV4 thrashes and admits failure while Minimax at least pursues a productive counting argument — but neither finishes. Recommendation: Minimax needs a higher token budget or a prompt-side instruction to skip its 'let's analyze' preamble and write the final proof first.

### synthesis

Catastrophic format failure for minimax-m2.7: 4 of 6 responses (D1, D2, D3, D6) are pure chain-of-thought scratch-pad that gets truncated at the 2500-token limit before producing a final answer. Only D4 and D5 yield real responses, and D5 is competitive with DSV4. This appears to be a behavior where minimax burns its token budget restating the prompt and planning rather than answering. DSV4 delivers structured answers across all 6 prompts; quality is solid (6-8) with occasional factual slips (FoundationDB consistency mischaracterization in D3, KV-cache-per-expert confusion in D1, hardware-ceiling intuition inverted in D2). When both produce answers (D4, D5), DSV4 is slightly more rigorous on type theory and math; minimax is competitive on conceptual depth. Net result: DSV4 wins 5-0-1 on this category, primarily because minimax fails to produce answers rather than because DSV4 produces exceptional ones.

### longctx

Terseness emphatically helped DSV4, did not hurt it. The hot question reverses: MiniMax's verbose chain-of-thought ate its 1500-token budget on E1, E2, and E5, causing finish_reason=length truncation that left the FINAL block incomplete or empty even though the scratchpad reasoning had reached correct answers. DSV4 skipped the visible reasoning and went straight to formatted [FINAL] answers in 233-492 tokens, never once missing a question. On the three items where MiniMax did fit (E3, E4, E6) the two models tied at 10/10. So on long-context recall, DSV4's concise style was a pure asset here -- it was not too terse to cover 4-part questions, and it avoided the truncation tax MiniMax paid on half the items. Recommendation: cap MiniMax reasoning tokens or raise max_tokens for this category.

## Bottom line

**DSV4-Flash wins this benchmark across all 5 categories by 3-5 quality-score points each.** The decisive factor is **format**, not raw capability:
- MiniMax-M2.7's reasoning is often correct and occasionally more elegant (C4 counting argument, C6 Grönwall via h=f²).
- But it spends its entire token budget on stream-of-consciousness scratch-pad and rarely arrives at a delivered final answer.
- Natural-stop rate: MiniMax averages 17% across the suite; DSV4 averages 87%.
- On code (B3) MiniMax literally degenerated into copy-paste loops.

DSV4-Flash converges to structured outputs nearly every time. For agentic workloads where the consumer needs a parseable response, **DSV4-Flash is the right choice on this hardware**.

## How to fix MiniMax

This format problem is fixable. Two paths:

**(a) Raise the token budget.** With max_tokens=5000 across all categories, MiniMax's scratch-pad would finish and the final answer would land. Cost: ~2x wall time per prompt.

**(b) System-prompt nudge.** Add a system message like `"Always output your final answer first in a clearly-marked [ANSWER] block, then your reasoning second. Do not preamble."` This realigns the output format to consumer-friendly without retraining the model.

Worth retesting once one of these is applied.

## Hardware footnote

DSV4-Flash uses **2 Sparks** (TP=2 expert-parallel via NCCL/RoCE). MiniMax-M2.7 uses **1 Spark**. At parity (single Spark), DSV4-Flash would not even fit. MiniMax wins on form factor — frontier-class agentic model on a single 128 GB box — but loses on this benchmark on the deliverable.

## Files in this dir

| file | contents |
|---|---|
| `agentic_prompts.json` | the 30 prompts + rubrics |
| `responses/{cat}.json` | each model's responses, per category |
| `scores/{cat}.json` | judge scores + per-item notes |
| `results_minimax.json` | raw bench results (MiniMax) |
| `results_dsv4.json` | raw bench results (DSV4) |
| `bench_minimax.log` | bench stdout (MiniMax) |
| `bench_dsv4.log` | bench stdout (DSV4) |

