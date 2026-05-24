# Hard Agentic Bench v1 -> v2 -> v3: MiniMax-M2.7 on DGX Spark

Three-iteration evaluation against same 30-prompt hard agentic suite. DSV4-Flash (dual-Spark TP=2) v2 results used as comparison baseline (reasoning toggle doesn't apply to vLLM, so DSV4 unchanged across rounds).

## Intervention recipe (cumulative)

| | v1 (default) | v2 (+ system prompt + bigger budget) | v3 (+ reasoning off) |
|---|---|---|---|
| System prompt | (none) | "FINAL block first, reasoning after, no preamble" | same as v2 |
| max_tokens | 1500-3000 | 5000-6000 | same as v2 |
| Server flag | `--reasoning auto` (default) | same as v1 | **`--reasoning off --reasoning-budget 0`** |
| Behavior | MiniMax burns budget on `<think>` | bigger budget; sometimes delivers | no thinking; goes straight to content |

## Headline quality trajectory

| category | MM v1 | MM v2 | **MM v3** | DSV4 |
|---|---:|---:|---:|---:|
| planning | 3.00 | 5.00 | **6.92** ↑ | 6.33 |
| code | 1.00 | 2.83 | **3.67** ↑ | 4.00 |
| math | 4.67 | 7.00 | **6.83** ↓ | 7.50 |
| synthesis | 3.50 | 4.33 | **6.17** ↑ | 6.67 |
| longctx | 7.83 | 8.92 | **9.75** ↑ | 10.00 |
| **OVERALL** | **4.00** | **5.62** | **6.67** | **6.90** |

## Wins (head-to-head per item)

| category | v1 | v2 | **v3** |
|---|---|---|---|
| planning | DSV4 6 - MM 0 - tie 0 | DSV4 5 - MM 1 - tie 0 | **DSV4 1 - MM 5 - tie 0** |
| code | DSV4 6 - MM 0 - tie 0 | DSV4 4 - MM 1 - tie 1 | **DSV4 3 - MM 2 - tie 1** |
| math | DSV4 5 - MM 1 - tie 0 | DSV4 4 - MM 1 - tie 1 | **DSV4 3 - MM 0 - tie 3** |
| synthesis | DSV4 5 - MM 0 - tie 1 | DSV4 6 - MM 0 - tie 0 | **DSV4 2 - MM 0 - tie 4** |
| longctx | DSV4 3 - MM 0 - tie 3 | DSV4 2 - MM 0 - tie 4 | **DSV4 1 - MM 0 - tie 5** |
| **TOTAL** | DSV4 25 - MM 1 - tie 4 | DSV4 21 - MM 3 - tie 6 | **DSV4 10 - MM 7 - tie 13** |

## Format / efficiency (MiniMax)

| | v1 | v2 | **v3** |
|---|---:|---:|---:|
| Total wall | 56.9 min | 118.0 min | **48.7 min** |
| Total output tokens | 70,593 | 133,115 | **59,444** |
| Natural-stop rate | 5/30 | 9/30 | **29/30** |
| Mean decode | 20.49 t/s | 18.51 t/s | **19.97 t/s** |

v3 is **2.4x faster** than v2 and uses **55% fewer tokens** while producing **higher-quality output**.

## Per-category v3 takeaways

### planning

v1 -> v2 -> v3 MiniMax: 3.0 -> 5.0 -> 6.92 (+3.92 cumulative). v1 -> v2 -> v3 DSV4: 6.83 -> 6.33 -> 6.33 (unchanged in v3 since DSV4 has no reasoning toggle). Wins flipped: v1 was DSV4 6-0, v2 was DSV4 5-1, v3 is MiniMax 5-1. The reasoning-off intervention is the clear winner for planning-category tasks: MiniMax now jumps directly to a structured [FINAL] block, doesn't burn 15-19k chars on self-narration, and the 5000-token budget is spent on the deliverable instead of soliloquy. All 6 responses are finish=stop (vs 1 of 6 in v2). Token efficiency: v2 MiniMax averaged ~5000 tokens (max) with truncation; v3 averages ~2783 tokens with completion — quality went UP while output shrank ~44%. The one regression is A5 (the hardware-spec question): in v2, MiniMax's reasoning chain caught the 1-GPU-per-Spark fact that DSV4 missed; in v3 with reasoning OFF, MiniMax now makes the SAME hardware mistake as DSV4 (hallucinated multi-GPU per node, MiniMax even goes further with TP=4 vs DSV4's TP=2). This is a real cost: reasoning-off trades subtle factual recall on niche hardware questions for massive gains in instruction-following and output discipline. Net: reasoning-off is a clear win for 5 of 6 planning tasks; the loss on A5 is the price paid. Recommendation: keep reasoning OFF for planning/architecture/diagnostic prompts; consider keeping it ON for fact-dense questions about specific hardware SKUs where the reasoning chain catches errors a non-reasoning pass would commit. The fundamental v2 bottleneck (output-budget discipline) has been solved; the new v3 limitation is occasional factual collapse on niche specs.

### code

v3 (reasoning OFF) moves minimax from 2.83 (v2) to 3.67 — closing the gap with DSV4 from 1.17 points to 0.33 points. The trajectory is v1: 1.0 (no compilable code at all) -> v2: 2.83 (emits code but 3/6 truncated, often buggy) -> v3: 3.67 (5/6 terminate cleanly, real code shipped). Token usage dropped substantially: v2 frequently hit 6000-token cap; v3 totals are 3649/1582/1908/6000/936/2145 — only B4 still hits the cap, and even there the truncation falls on the test summary printf rather than the actual function (v2 truncation was always inside the implementation). The 'less verbose deliberation, more actual code' hypothesis is CONFIRMED for delivery completeness. Wins explicitly traceable to v3: B4 went from 2 -> 5 (now ships a working json_extract_int + 20 tests; v2 had only helpers and no target function); B5 went from 3 -> 4 (now identifies the id(object()) rubric bug exactly, which v2 entirely missed; correct lock answer). The 'no reasoning leading to bugs' worry IS borne out for B1 and B3: B1 ships a Shard struct without the RwLock field it claims to use (a reasoning pass would have caught this trivially) — score stays at 2. B3 REGRESSED (5 -> 4) because v3 lost the redis.call('TIME') compliance that v2 had, and introduced a silent-mutation bug where remaining/resetAt accidentally execute the write-script SHA. So reasoning-off is a net win for length-budget completion (B4, B6) and for hitting the obvious rubric items concisely (B5), but it loses sanity checks that catch trivial type-system and rubric-compliance errors (B1, B3). Wins now 2-3-1 (minimax-dsv4-tie) vs v2's 1-4-1 — minimax doubled its outright wins. Next lever: combine reasoning-on with a target-shape system prompt that forces a final-pass compile-check / rubric-checklist before [FINAL], or run reasoning-on only for the harder items (B1, B2, B3) and reasoning-off for the ones that just need delivery (B4, B5, B6).

### math

REASONING-OFF (v3) is a MIXED result on math, and the prediction held: math IS the category most exposed when internal thinking is disabled. Structural wins from v3: every single MiniMax response now finishes with finish_reason=stop (no truncation), uses fewer tokens (avg ~1525 vs v2 hitting 5000 ceiling repeatedly), and produces a complete-looking polished artifact for every problem. Substantive losses from v3: the answers contain NEW math errors that v2 did not have, specifically the kind that internal step-by-step verification would have caught. C1 v3 asserts a wrong counterexample (x=-2,n=2 — actually holds) and a wrong failure-set characterization (all odd n — false for x=-2,n=3); v2 had truncation but the partial content was correct. C4 v3 invents a bogus Sylow case (n_3=10 doesn't divide 25), miscounts groups of order 81 as 7 (correct is 15), arrives at total 15 (correct is 30); v2 also failed C4 but failed via truncation, not via false statements. C5 v3 has an uncorrected arithmetic slip (2(n+1)H_n - 2n is what the algebra gives; magic-words it to -4n) that v2 did not have. C6 v3 INVERTS the sharp threshold in (b), claiming alpha < 1 forces f=0 (the exact opposite of truth, and contradicting its own (a) counterexample) — v2 truncated but never made this assertion. Net scoring: avg went from 7.0 (v2) to 6.83 (v3), a small regression of 0.17 points. WHY MATH SUFFERED: proofs are exactly the artifact type where 'think internally first, then write' beats 'write the proof while thinking on paper': the internal pass catches sign errors, divisibility-mod errors, threshold-inversions, and arithmetic slips before they reach the output. With reasoning off, MiniMax writes plausible-sounding LaTeX with internal logical errors. C2 and C3 are the exceptions (both improved to 9) because those problems either have a stable algebraic shortcut (C3 identity) or are guided by a strong unique answer (C2 c=1/(4sqrt 2)) that the model has likely memorized. C1, C4, C5, C6 — problems requiring fresh casework, counterexample construction, or threshold direction — all show new errors that v2 did not have. Score gap to DSV4 widens slightly from 0.5 (v2) to 0.67 (v3). RECOMMENDATION: for math specifically, reasoning-off is the wrong tradeoff — the truncation problem v2 prompt-engineering solved is better than the silent-arithmetic-error problem v3 introduces. For categories where the artifact is exposition rather than chain-of-correctness (synthesis, planning), reasoning-off may still be the right call; math wants reasoning back on with the v2-style prompt and a higher token budget. DSV4 essentially unchanged across all three versions (7.33 -> 7.5 -> 7.5), with the one wrinkle that C4 DSV4 v3 now truncates after thrashing for 5000 tokens (was always weak here).

### synthesis

v3 (--reasoning off) is a STRUCTURAL VICTORY for minimax. The scratch-pad-then-truncate failure mode is fully eliminated: all 6 responses now have natural stops with [FINAL] at character 0 or near 0, and the model spends 100% of its 5000-token budget on structured content instead of 80-95% on chain-of-thought preamble. Avg jumps 3.5 (v1) → 4.33 (v2) → 6.17 (v3). Win count goes 5-0-1 → 6-0-0 → 2-0-4 (DSV4-tie-minimax flipped to DSV4-minimax-tie). KEY QUESTION ANSWERED: dropping the reasoning DID NOT kill mechanism depth. Where v2's [FINAL] sections were 1000-3000 tokens of truncated structured content, v3's are 1500-3500 tokens of complete structured content with COMPARABLE per-section depth to what the scratch-pad versions produced internally. Examples: D1 v3 has the same bias-update formula and capacity-aware-clipping argument the v2 scratch-pad reasoned toward but couldn't deliver; D6 v3 has the same Ulysses all-to-all + concrete bandwidth arithmetic; D5 v3 retains the full PPO L^CLIP formula minimax was already good at. The depth/specificity is preserved — the model just no longer narrates the path getting there. WHAT IS LOST: a small number of mid-reasoning hedges and self-corrections that the v2 scratch-pad exposed. v3 makes its mistakes more confidently (CockroachDB default = SI in D3 is now stated assertively instead of revisited; hardware ceilings for spec-decoding still inverted; KV-cache-per-expert mechanism error in D1). The 'reasoning off' mode produces more polished surface text but loses the v2 model's ability to internally catch its own errors. NET ASSESSMENT: this is the correct tradeoff for this benchmark. Going from 0/6 complete answers (v1) to 0/6 complete answers (v2 — all truncated) to 6/6 complete answers (v3) is the dominant axis of improvement. The remaining gap to DSV4 is now ABOUT CONTENT, not about structure: minimax v3 still has (i) the CRDB-default-isolation error (D3, persistent across v2→v3), (ii) inverted hardware-ceiling intuition for spec decoding (D2, shared with DSV4 — not a differentiator), (iii) mechanism errors on per-expert KV cache (D1), (iv) wrong claim that Megatron has no native SP (D6), (v) confabulated paper title for the RLHF failure-mode citation (D5), (vi) missed ARC-as-runtime-cost rubric item (D4). DSV4 still wins on D1 (mechanism accuracy) and D4 (ARC + concrete Swift example), but FOUR ties is a real shift — v2 had ZERO ties, v3 has FOUR. With one more iteration fixing the CRDB-default error and the per-expert-cache mechanism error, minimax would plausibly start winning categories outright. THE FIX SUGGESTED IN v2 NOTES (compress scratch-pad / disable pre-FINAL reasoning) was implemented as --reasoning off and WORKED. Trajectory: v1 3.50 (sweep loss) → v2 4.33 (sweep loss, structural failure) → v3 6.17 (tie-heavy, structural parity).

### longctx

v3's --reasoning off on the MiniMax server is the highest-leverage single change in the longctx sweep. The pathological case (E5) that burned 3500 tokens on a self-referential formatting meta-loop in both v1 and v2 is GONE: v3 delivers a clean 626-token FINAL block with all 4 answers correct (5->10, +5 pts single-handed). E1 remains the lone non-tie: MiniMax still selects the wrong 'maximum p99' on Q4 -- v1 had no FINAL at all, v2 picked 4.2s (the first-impact telemetry), v3 picks 9.1s (the post-mitigation peak), but the rubric's canonical answer is the IMPACT section's 'above 15 seconds.' This is a content-extraction bias, not a reasoning-budget issue, and reasoning-off doesn't help. E2/E3/E4/E6 stay at 10/10 ties. Net trajectory: MiniMax longctx avg 7.83 -> 8.92 -> 9.75 (+1.92 total), wins ledger 0-3-3 -> 0-2-4 -> 0-1-5. DSV4 holds perfect 10.0 across all three rounds. With reasoning-off, MiniMax recall did NOT degrade -- the terser outputs (279-626 tokens vs 1000+ in prior rounds) extract the same answers cleanly. The remaining 0.25-pt gap is one specific content-extraction error on E1's Q4 max-p99 (picking timeline peaks over IMPACT-section summary). To close: instruct the model to prefer summary/IMPACT sections over event-timeline peaks when asked for 'maximum,' or include this as a rubric clarification in the prompt.

## The big picture

**`--reasoning off --reasoning-budget 0` is the single most decisive intervention** for MiniMax-M2.7 on agentic workloads. The v1->v2 fix (system prompt + raised budget) gave MiniMax room to deliver but didn't change behavior — it still burned tokens on stream-of-consciousness scratch-pad. v3 strips that out entirely at the server level: the model is forced to produce final-form content directly.

**Quality** climbed in every category except math:

| category | v1->v3 change |
|---|---:|
| planning | **+3.92** (3.00 -> 6.92, **now beats DSV4**) |
| code     | **+2.67** (1.00 -> 3.67, gap closed from 3.17 -> 0.33) |
| synthesis| **+2.67** (3.50 -> 6.17, gap closed from 3.17 -> 0.50) |
| longctx  | **+1.92** (7.83 -> 9.75, gap closed from 2.17 -> 0.25) |
| math     | **+2.16** (4.67 -> 6.83, but v3 < v2 by 0.17 - reasoning-off introduced new arithmetic errors) |

**Overall**: MiniMax went from **4.00 -> 5.62 -> 6.67** while DSV4 stayed at ~6.90. The gap closed from **2.90 points to 0.23 points**.

**Wins ledger v1->v3**: DSV4 25-1-4 -> 21-3-6 -> **10-7-13**. MiniMax went from 1 win in 30 to **7 wins**.

**Wall time v1->v3**: MiniMax 56.9 -> 118.0 -> **48.7 min**. v3 is faster than v1 AND v2 because reasoning-off cuts output volume by half.

## Where MiniMax now beats DSV4 outright

**Planning category, v3**: MiniMax 6.92 > DSV4 6.33. MiniMax won 5 of 6 items (A1, A2, A3, A4, A6). DSV4 only won A5 - the hardware-specific question where MiniMax confidently hallucinated 4xNVLink-C2C on the single-GPU GB10. v2's reasoning chain was catching that error; v3's reasoning-off let it ship.

## The math trade-off

Math is the one category where reasoning-off was a net loss. v2 with reasoning-on had MiniMax catch its own arithmetic errors mid-scratch-pad (Bernoulli inequality counterexample, Sylow divisibility, sharp-threshold direction). v3 ships those errors directly without self-correction. The math judge flagged 4 new errors: C1 wrong counterexample asserted, C4 Sylow undercount (7 vs 15 groups), C5 algebra hand-wave, C6 inverted sharp threshold.

**Recommendation**: route math/proof workloads to a separate llama-server instance with `--reasoning auto`, and everything else to `--reasoning off`. Two ports at the gateway level (e.g., 8096 reasoning-off default, 8097 reasoning-auto for math) handle this cleanly.

## Status now

MiniMax-M2.7 live at `http://10.117.1.215:8096/v1` with `--reasoning off --reasoning-budget 0`. SparkLinus configured to use this endpoint as default.

## Files

```
AGENTIC_BENCH.md          # v1 report
AGENTIC_BENCH_V2.md       # v2 report (v1 vs v2)
AGENTIC_BENCH_V3.md       # this file (v1 vs v2 vs v3)
results_minimax{,_v2,_v3}.json   # raw results
results_dsv4{,_v2}.json
scores{,_v2,_v3}/{cat}.json      # per-category judge scores
responses{,_v2,_v3}/{cat}.json   # responses sent to judges
run_agentic_bench{,_v2}.py       # bench harnesses
agentic_prompts.json             # the 30 hard prompts
launch_minimax_m27_noreason.sh   # v3 launcher (--reasoning off)
```