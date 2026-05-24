# MiniMax-M2.7 UD-IQ4_XS → DGX Spark (single node)

Single-Spark deployment of `unsloth/MiniMax-M2.7-GGUF` at the **UD-IQ4_XS**
quant — the largest of unsloth's Dynamic Imatrix quants that fits within
the Spark's effective memory budget.

## Why this quant

| Variant | Size | Fits 121 GB usable? |
|---|---:|---|
| UD-IQ3_XXS | 80.1 GB | ✅ comfy |
| UD-Q3_K_M | 101.2 GB | ✅ comfy |
| UD-Q3_K_XL | 101.9 GB | ✅ comfy |
| **UD-IQ4_XS** | **108.4 GB** | **✅ ~13 GB headroom for KV** |
| UD-IQ4_NL | 110.8 GB | ⚠ very tight |
| UD-Q4_K_S | 131.0 GB | ❌ overflow |

128 GB hardware → 127.6 GB kernel-addressable (MemTotal) → ~121 GB realistically
usable by a single process after kernel/firmware reserves. UD-IQ4_XS is the
**best quality quant that fits with room for a 32K KV cache**.

## Architecture notes

- MiniMax-M2.7 is **230B-A10B MoE** (~230B total params, ~10B active per token)
- Decode rate is bandwidth-limited: ~10 GB active read per token × 273 GB/s LPDDR5X = ceiling ~27 t/s
- llama.cpp build 9304+ on the Spark supports MiniMax-M2 arch natively
- llama.cpp loads multi-shard GGUFs automatically when given the first shard

## Files

```
launch.sh                  # llama-server launcher (port 8096, foreground)
llama-minimax27.service    # systemd unit (multi-user)
smoke_bench.sh             # health + quick 5-prompt perf bench
README.md                  # this file
```

## Optimizations (vs stock defaults)

| Flag | Value | Why |
|---|---|---|
| `-ngl 999` | full GPU/unified-memory offload | Spark unified memory = all VRAM |
| `-c 32768` | 32K context | leaves ~13 GB free for KV; bump only if `free -h` allows |
| `-b 512 -ub 256` | prefill batch shape | wider ubatch than 27B because MoE prefill is cheaper per token |
| `-fa on` | FlashAttention | required for memory-efficient long ctx |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache | half-precision KV, ~3 GB at 32K |
| `--parallel 1` | single slot | low TTFT; raise to 4-8 only after monitoring memory |
| `--cache-ram 0 --no-cache-prompt --ctx-checkpoints 0` | clean memory budget | avoid host-RAM swap (no swap on unified mem) |
| `--mlock` | lock pages | prevent paging during decode |
| `--jinja` | use model chat template | MiniMax ships one |
| `--reasoning auto --reasoning-format deepseek --reasoning-budget 4096` | thinking model support | M2.7 is reasoning-capable |
| `--threads 12` | host worker pool | Spark has 20 CPU cores |

## Deploy

On spark-2 as `sunil`:

```bash
# 1. wait for download (~60-90 min @ ~1.7 GB/min for 108 GB)
ssh sunil@10.117.1.215 'du -sh /home/sunil/models/MiniMax-M2.7/UD-IQ4_XS/'
# expected: 108G after completion

# 2. copy launcher
scp launch.sh sunil@10.117.1.215:/home/sunil/launch_minimax_m27.sh
ssh sunil@10.117.1.215 'chmod +x /home/sunil/launch_minimax_m27.sh'

# 3. start (foreground first, watch logs)
ssh sunil@10.117.1.215 'bash /home/sunil/launch_minimax_m27.sh 2>&1 | head -200'
# wait for "main: server is listening on http://0.0.0.0:8096"

# 4. smoke + bench from your Mac (or anywhere on Tailscale)
bash smoke_bench.sh http://100.117.56.97:8096

# 5. install as service once smoke passes
scp llama-minimax27.service sunil@10.117.1.215:/tmp/
ssh sunil@10.117.1.215 'sudo cp /tmp/llama-minimax27.service /etc/systemd/system/ && \
                       sudo systemctl daemon-reload && \
                       sudo systemctl enable --now llama-minimax27.service'
```

## Expected performance

Based on bandwidth roofline (10 GB active / 273 GB/s = 27 t/s no-spec ceiling):

| Workload | Expected decode t/s |
|---|---:|
| Short Q&A (high MoE locality) | 22-27 |
| Code generation | 18-23 |
| Long-form (KV grows) | 15-20 |
| Random worst-case | 10-14 |

If anything significantly under these, suspect:
- `--threads` too low (try 16)
- `-ngl` not actually offloading everything (check `nvidia-smi` during decode)
- Reasoning blocks padding output (set `--reasoning-budget 1024` to cap)

## Tuning levers

| Want | Change |
|---|---|
| Longer context (64K) | `CTX=65536 BATCH=256 UBATCH=128 bash launch.sh`. Watch RSS — may overflow. |
| Lower memory pressure | Drop to `UD-Q3_K_XL` (101 GB) — frees 7 GB for bigger KV |
| Higher quality, ditch reasoning | Move to `UD-Q4_K_XL` (140 GB) — won't fit; needs dual-Spark RPC |
| Dual-Spark RPC | llama.cpp's `--rpc` to split weights across both Sparks (advanced; future work) |

## Routing integration

Add to the Spark's gateway config (`remote-work/config.yaml` equivalent):
```yaml
  base_url: http://100.117.56.97:8096/v1
  model: minimax-m2.7
```

## Measured (live)

Deployed 2026-05-24 on spark-2 (10.117.1.215:8096), llama.cpp build 9304.

5-prompt T=0 bench, 600-token cap, single-stream c=1:

| | t/s |
|---|---:|
| Mean decode | **23.34** |
| Median | 23.36 |
| Peak | 23.54 |
| Min | 23.03 |
| Prefill | 117-121 |
| TPOT | 42.5-43.0 ms |

Variance <0.5 t/s — extremely consistent. ~86% of the bandwidth roofline ceiling
(10 GB active × 273 GB/s LPDDR5X = ~27 t/s no-spec).

## Max-context push (2026-05-24)

Bumped from 32K → **196608 (192K)** by switching KV cache from q8/q8 → q4_0/q4_0.

| | 32K q8/q8 | **192K q4_0/q4_0** | Δ |
|---|---:|---:|---|
| ctx | 32,768 | **196,608** | **6×** |
| KV @ full | 4.2 GB | 12.4 GB | +3× |
| total resident | 117 GB | **120.5 GB** | +3% |
| mean decode | 23.34 t/s | **23.55 t/s** | **+1%** |
| peak decode | 23.54 t/s | **23.92 t/s** | +1.6% |
| TPOT | 42.7 ms | 42.5 ms | -0.5% |

**Free win**: 6× context unlocked, zero speed regression. q4_0 KV at single-stream
short prompts has same per-step cost as q8 (active KV reads are tiny — cost
shows up only at near-full ctx, where the long-prompt prefill dominates anyway).

192K = MiniMax-M2.7's training-time maximum (`n_ctx_train` in the GGUF). The
model card lists 1M but that requires YaRN extrapolation; the GGUF itself
caps at 192K. Going past that requires `--yarn-orig-ctx 196608 --rope-scaling yarn`
+ a scaling factor — quality at scaled ctx is worse than at native.

### Live config

```
CTX=196608  KTYPE=q4_0  VTYPE=q4_0  bash launch_maxctx.sh
```

Same flags as `launch.sh`, just parameterized. Memory used: 120 GB / 121 GB usable
(tight but stable). On crash/OOM: drop to `CTX=131072` for safe 128K + 4 GB headroom.

### Verified at 4K real prompt

Long-context needle-in-haystack with the new 192K cache:
- prefill 385 t/s, decode 17.55 t/s (decode slows at long ctx due to attention)
- correctly extracted `QUARTZ-VIOLET-7331` from filler-buried needle
