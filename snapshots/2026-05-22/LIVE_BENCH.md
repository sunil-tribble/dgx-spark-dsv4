# Live bench — 2026-05-22 14:31 UTC

Captured immediately after the dual-Spark cluster came up post power-cycle.

## Cluster configuration

| | |
|---|---|
| vLLM | jasl/vllm @ dda4668b5 (rowwise paged-MQA logits for SM12x) |
| Image | vllm-node-dsv4:latest |
| Model | DeepSeek-V4-Flash-abliterated-v3 (served as "deepseek-v4-flash") |
| Topology | TP=2, EP enabled, nnodes=2, master=10.117.1.215:29501 |
| Backend | FLASHINFER_MLA_SPARSE, FP8 KV, prefix caching on |
| Spec | deepseek_mtp num_speculative_tokens=3 |
| Context | max_model_len=1048576, max_num_batched_tokens=32768 |
| Batch | max_num_seqs=1 (single-stream eval) |

## Cold short prompts (~34 in / 500 out, T=0.7)

```
run 1: prompt=34 out=500 time=28.12s decode=17.78 tok/s
run 2: prompt=34 out=500 time=26.30s decode=19.01 tok/s
run 3: prompt=34 out=500 time=26.87s decode=18.61 tok/s
run 4: prompt=34 out=500 time=28.89s decode=17.31 tok/s
run 5: prompt=34 out=500 time=29.18s decode=17.14 tok/s

AVG  17.97 tok/s   PEAK 19.01 tok/s
```

## Warm multi-turn (3 turns, growing context, prefix-cache warmed)

```
turn 1: prompt=38   out=582 time=32.68s decode=17.81 tok/s
turn 2: prompt=633  out=600 time=31.22s decode=19.22 tok/s
turn 3: prompt=1251 out=600 time=30.94s decode=19.39 tok/s

WARM AVG 18.81 tok/s   PEAK 19.39 tok/s
```

## MTP spec decode acceptance (live counters)

```
total drafts             3047
total draft tokens       9141   (= 3 per draft, confirms spec=3)
total accepted tokens    4045

per-position acceptance:
  pos 0: 2404 / 3047 = 78.9%   (canonical ~84%)
  pos 1: 1246 / 3047 = 40.9%   (canonical ~50%)
  pos 2:  395 / 3047 = 13.0%   (canonical ~16%)

effective token expansion: 1 + 4045/3047 = 2.33x
realized decode speedup over no-spec baseline: ~1.3x
```

## Cache behavior

```
prompt_tokens_total       2318
  local_cache_hit         1536  (66%, from prefix cache)
  local_compute            782
generation_tokens_total   7096
```

## Interpretation

The measured 18-19 tok/s is roughly half the canonical 35 t/s "Conv MTP=2 c=1
mt-bench" number. The gap matches the existing repo's documented "verifier
overhead dominates" finding for spec=3 on DSV4-Flash — theoretical speedup
from 2.33x token expansion is only realized as ~1.3x due to per-step
verifier cost.

To approach 30+ tok/s, the documented levers are:
1. Use the harness's Conv profile (max-num-seqs 36, max-num-batched-tokens
   8192, max-model-len 32768) — this is what the 35 t/s number was measured on
2. Reduce to spec=2 (drop pos 2 which has only 13% acceptance — net
   verifier cost outweighs gain)
3. Run with the actual jasl/vllm-ds4-sm120-harness dataset (mt-bench
   prompts), not synthetic ones

What we've validated:
- ✅ Cluster comes up cleanly from power-cycle in ~10 min
- ✅ Both nodes communicate over RoCE (NCCL_IB_HCA discovery worked)
- ✅ Worker connects to head's master port and serves rank 1
- ✅ FLASHINFER_MLA_SPARSE backend active
- ✅ MTP-3 draft → verify path works, acceptance matches documented profile
- ✅ FP8 KV cache active, 17.87 GiB allocated per worker
- ✅ Prefix caching cuts prefill on warm turns (66% hit rate observed)
