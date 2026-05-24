# Hard Agentic Bench v1 vs v2: MiniMax-M2.7 vs DeepSeek-V4-Flash

## What changed in v2

Same 30 prompts. Same hardware (single Spark for MiniMax, dual-Spark TP=2 for DSV4-Flash). Two interventions applied to **both** models for fair comparison:

1. **System prompt nudge** (new in v2):
   > "Output your FINAL deliverable answer FIRST in a clearly-marked [FINAL] block. Put any reasoning, alternative considerations, or scratch work AFTER the [FINAL] block. Do not preamble. Do not narrate what you are about to do. Start with [FINAL] immediately."

2. **Raised max_tokens** per category:

| category | v1 cap | v2 cap |
|---|---:|---:|
| planning | 2500 | **5000** |
| code | 3000 | **6000** |
| math | 2500 | **5000** |
| synthesis | 2500 | **5000** |
| longctx | 1500 | **3500** |

## Headline: quality score deltas

| category | MM v1 → v2 | Δ | DSV4 v1 → v2 | Δ |
|---|---:|---:|---:|---:|
| planning  | 3.00 → 5.00 | **+2.00** | 6.83 → 6.33 | -0.50 |
| code      | 1.00 → 2.83 | **+1.83** | 4.17 → 4.00 | -0.17 |
| math      | 4.67 → 7.00 | **+2.33** | 7.33 → 7.50 | +0.17 |
| synthesis | 3.50 → 4.33 | **+0.83** | 6.67 → 6.67 | +0.00 |
| longctx   | 7.83 → 8.92 | **+1.09** | 10.00 → 10.00 | +0.00 |
| **OVERALL** | **4.00 → 5.62** | **+1.62** | **7.00 → 6.90** | -0.10 |

## Wins by category (v2)

| category | DSV4 wins | MiniMax wins | ties | v1 result |
|---|---:|---:|---:|---|
| planning  | 5 | 1 | 0 | DSV4 6 - MM 0 - tie 0 |
| code      | 4 | 1 | 1 | DSV4 6 - MM 0 - tie 0 |
| math      | 4 | 1 | 1 | DSV4 5 - MM 1 - tie 0 |
| synthesis | 6 | 0 | 0 | DSV4 5 - MM 0 - tie 1 |
| longctx   | 2 | 0 | 4 | DSV4 3 - MM 0 - tie 3 |
| **OVERALL** | **21** | **3** | **6** | DSV4 25 - MM 1 - tie 4 |

## Speed (v2)

| | MiniMax-M2.7 (1 Spark) | DSV4-Flash (2 Sparks TP=2) |
|---|---:|---:|
| Total wall time | 118.0 min | **26.0 min** |
| Total output tokens | 133,115 | 38,654 |
| Mean decode t/s | 18.51 | **23.52** |
| Natural-stop rate (across all 30) | 30% | 97% |

## Per-category speed + format

| category | MM decode | MM out | MM stop% | D4 decode | D4 out | D4 stop% |
|---|---:|---:|---:|---:|---:|---:|
| planning  | 19.04 t/s | 4998 | 17% | 23.08 t/s | 1394 | 100% |
| code      | 18.51 t/s | 5704 | 50% | 26.26 t/s | 1729 | 100% |
| math      | 19.25 t/s | 5000 | 0% | 27.94 t/s | 2018 | 83% |
| synthesis | 19.18 t/s | 5000 | 0% | 22.60 t/s | 1062 | 100% |
| longctx   | 16.56 t/s | 1484 | 83% | 17.72 t/s | 238 | 100% |

## Per-category v2 takeaways

### planning

v1 -> v2: MiniMax 3.0 -> 5.0 (+2.0), DSV4 6.83 -> 6.33 (-0.5). The [FINAL] intervention PARTIALLY worked: 6 of 6 MiniMax responses now contain a [FINAL] tag (vs 0 of 6 in v1), and A5 produced a complete FINAL...[/FINAL] block that actually beat DSV4 on a hardware-spec question. BUT MiniMax still does not put [FINAL] FIRST as instructed - it writes 15-19k chars of reasoning/narration before emitting [FINAL], which means 5 of 6 responses hit max_tokens=5000 and truncate mid-FINAL. The 5000-token cap is being burned on reasoning preamble rather than the deliverable. Two follow-up interventions would close the gap: (a) stronger instruction-following discipline so [FINAL] comes immediately (not after a 'we need to...' soliloquy and not duplicated 9 times as in A3); (b) raise max_tokens to 8000+ since even when MiniMax does emit FINAL, its FINAL blocks are denser/longer than DSV4's. MiniMax's underlying technical content is competitive (A5 win, A3 cryptographic specificity, A1 real pglogical functions) - the bottleneck is output-budget discipline, not knowledge.

### code

v2 system-prompt nudge + 6000-token cap unlocks code generation from minimax: in v1 it emitted ZERO compilable code (avg 1.0); in v2 it emits [FINAL]-tagged code on all 6 prompts and even wins B5 outright and ties B3. The improvement is structural — minimax now terminates deliberation and commits to a deliverable. But three of six responses (B1, B4, B6) still hit length=6000 mid-code with the actual deliverable truncated (no closing braces, missing the target function in B4, missing helper methods in B6). The committed code also has serious algorithmic bugs (B1 get doesn't update recency, B2 augmenting-path broken). DSV4 still wins 4-1-1 because partial-but-running code beats truncated code in head-to-head, but the gap has narrowed dramatically: from 4.17 vs 1.0 (3.17 point gap) to 4.0 vs 2.83 (1.17 point gap). The remaining ceiling on minimax is the 6000-token budget — three of six prompts need more headroom, and the deliberation-to-code ratio is still ~3:1 to ~4:1 even when it does terminate. Increasing max_tokens to 10000+ or further compressing the pre-FINAL deliberation are the obvious next levers.

### math

v2 prompt instructions ([FINAL]-block-first + 5000 token cap) WORK. Minimax improvement is dramatic: avg 4.67 -> 7.0. The biggest wins: C2 (4->8) where Minimax now gets c = 1/(4 sqrt 2) correct (v1 had algebra error giving 1/(8 sqrt 2)); C5 (5->9) where Minimax now delivers a complete polished proof of all four parts (was empty post-[FINAL] in v1); C3 (6->8) where the great-circle insight is now in a final artifact; C1 (3->6) where induction proof + (a) + (b) are now delivered (though c,d still truncated). C4 (5->5) unchanged because the underlying group-theory difficulty isn't a packaging issue. C6 (5->6) main proof now landed but part (b) still truncates. Score gap closed from 5-1 / 2.67-point lead to 4-1-1 / 0.5-point lead. DSV4 still wins on overall coverage because Minimax STILL hits the 5000-token cap and now loses content on the back end of polished proofs (parts c/d of multi-part problems are the casualties) rather than the front end (scratch-work overhead). Recommendation for v3: either (i) further compress the meta-preamble by instructing 'no scratch reasoning before [FINAL], deliver the polished proof immediately and ONLY' (note Minimax still emits a long scratch-pad before its post-[FINAL] block in v2), or (ii) raise the cap to ~8k tokens to let the polished proofs complete. Path (i) is preferable since Minimax's scratch-pad still consumes ~60-80% of its budget.

### synthesis

v2 fixes the most catastrophic failure mode (4/6 non-responses in v1 become 6/6 partial responses in v2 with [FINAL] markers and actual structured content) but uncovers a NEW failure mode: minimax burns 80-95% of the 5000-token budget on scratch-pad reasoning before reaching the [FINAL] block, then produces 1000-3000 tokens of structured content that truncates mid-answer. Result: every minimax response is structurally incomplete — D1 missing (d), D2 covers 2.x of 5 methods and no final pick, D3 truncates mid-FoundationDB AND has a factual error (CockroachDB default = SI, wrong), D4 has a duplicated FINAL with truncation mid second version, D5 covers only (a)-(b-partial) of 5 sub-questions, D6 covers only (a) of 4 sub-questions. WHERE MINIMAX SHINES: when content is reached, it is technically excellent — D2 has the exact rejection-sampling acceptance formula DSV4 omits; D5 has the full PPO L^CLIP formula with three precise theoretical-loss reasons; D6's activation-memory analysis names ring-attention, Ulysses all-to-all, FSDP2 limitations correctly; D2's hardware-ceiling table is more detailed than DSV4's. NO HALLUCINATED SPECIFICS observed — all named techniques (Polonius, ARC, ~Copyable, %1 arrow, Ulysses, ring-attention, 1F1B, EAGLE, Medusa, Jacobi, Paxos LWT, Raft, NLL) are real and used correctly. The single factual error is CockroachDB's default isolation in D3. Net result: DSV4 sweeps 6-0-0 on this round (worse for minimax than v1's 5-0-1), because minimax now produces partial answers that get judged on completeness rather than non-answers that get a flat 2. The fix needed is to compress scratch-pad — either via a stricter system prompt forbidding pre-FINAL reasoning, or by raising max_tokens to ~12k, or by using a model that doesn't externalize its chain-of-thought. v2 delta: minimax avg +0.83 (3.5 -> 4.33), DSV4 unchanged at 6.67, win count went from 5-0-1 to 6-0-0 (minimax lost the D5 tie because its v2 response is more incomplete than v1's was).

### longctx

v2's fixes (system prompt putting [FINAL] block first + max_tokens=3500) recovered E2 fully (5->10) and recovered E1 most of the way (7->8.5; lost 1.5 pts to a Q4 max-p99 selection error, not truncation). E5 remained pathological even with 3500 tokens: MiniMax got stuck in a self-referential meta-loop about formatting and emitted no FINAL block, identical 5/10 score to v1. E3/E4/E6 were already 10/10 in v1 and stayed there. DSV4 unchanged at perfect 10s across all six items -- its terse 161-309 token style was sufficient for every 4-part question. Net: MiniMax +1.09 on longctx avg, DSV4 still wins category outright at 10.0 vs 8.92. The remaining gap is now mostly E5's failure mode (reasoning meta-loop, not pure truncation) plus the Q4 max-vs-first-impact selection on E1. To close further: add explicit instruction to not deliberate about formatting in the system prompt, or use stop sequences to cap scratchpad length.

## Bottom line

The intervention worked, partially. MiniMax-M2.7's overall quality went from 4.00 to 5.62 (+1.62), closing the gap with DSV4 from 3.0 to 1.28 points.

**Specific wins** for MiniMax in v2:
- Math gap closed from 2.67 → 0.5 points. Algebra error from v1 C2 fixed. C5 now ships a complete proof. Even won/tied 1/1 category items.
- Code went from 0-6 sweep to 1-4-1 (won B5, tied B3).
- Planning produced [FINAL] blocks on 6/6 prompts (vs 0/6 in v1), won A5 outright.
- Longctx gap closed from 2.17 → 1.08 points; E2 fully recovered (5/10 → 10/10).

**Remaining MiniMax failure modes** (consistent across judges):
1. **Pre-FINAL preamble unfollowed**: Despite the system prompt instructing answer-first, MiniMax still emits 15-20k characters of "we need to analyze..." before reaching [FINAL]. The model is **trained to think externally** and the system prompt can't override that on a 4B-active-param MoE.
2. **Truncation now hits the END of the FINAL block** (instead of preventing it entirely): synthesis judge flagged all 6 items as structurally incomplete (missing parts c/d of multi-part questions).
3. **Synthesis category regressed by 1 point** (DSV4 went 5-0-1 → 6-0-0) — MiniMax's v1 D5 tie became a v2 loss because the answer now truncates mid-question.

**DSV4 was effectively unchanged by the v2 fix** (overall 7.00 → 6.97, within noise). It already converges to structured outputs.

## What v3 would need

Judges across categories converged on the same recommendation:

1. **Stricter prompt** forbidding any pre-FINAL reasoning (e.g., "Do not write any text before [FINAL]. If you need to think, put it AFTER the [/FINAL] tag.") — and possibly a stop sequence on `<think>` or a specific marker.
2. **Or raise max_tokens to ≥8000-12000** to let MiniMax's verbose-but-correct reasoning finish AND deliver a complete FINAL.
3. **Or disable reasoning at the server level**: relaunch llama-server with `--reasoning off` to force MiniMax to NOT emit `<think>` blocks at all.

Option 3 is the most decisive. It's a 10-min cluster restart.

## Total session footprint

- v1 MiniMax bench: 56.9 min
- v2 MiniMax bench: **118.0 min** (2× longer because of larger token caps)
- v1 DSV4 bench: 33 min
- v2 DSV4 bench: **26.0 min** (faster because system prompt + natural-stop)
- Cluster swaps (4 × ~10 min): 40 min
- 10 judge agents (5 v1 + 5 v2 in parallel): ~5 min real time per batch
- **Total ~4 hours** of compute against the live hardware.

