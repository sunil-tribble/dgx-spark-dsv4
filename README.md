# DeepSeek V4 Flash on 2× DGX GB10 Spark

Production deployment of an abliterated, 1M-context DeepSeek V4 Flash on two NVIDIA DGX GB10 Spark nodes connected via 200Gbps RoCE. Tensor-parallel across both nodes, MTP speculative decoding, FP8 KV cache.

## Capabilities

| Property | Value |
|----------|-------|
| Model | DeepSeek V4 Flash (abliterated, ~74 GB FP8) |
| Context length | 1,048,576 tokens (1M) |
| Throughput (long decode) | 28–32 tok/s |
| Throughput (FreeLinus agentic) | 20–30 tok/s warm; ~40 tok/s decode-only on warm prefix |
| Tool calling | Native (`deepseek_v4` parser) |
| Reasoning | Native thinking tokens (`<think>...</think>`) |
| Speculative decoding | DeepSeek MTP, 3 draft tokens |
| Spec acceptance | pos 0 ≈ 84%, pos 1 ≈ 50%, pos 2 ≈ 16% (≈2.5x speedup vs no spec) |

## Hardware

| Node | Role | IP (Tailscale) | IP (RoCE switch) | IP (P2P cable) |
|------|------|----------------|------------------|----------------|
| spark-2 | Head | 100.117.56.97 | 192.168.100.2 | 192.168.101.2 |
| spark-1 | Worker | 100.117.56.98 | 192.168.100.1 | 192.168.101.1 |

- **GPU**: NVIDIA GB10 (Grace Blackwell, SM12x) — 1 per node, unified memory with Grace CPU
- **Interconnect**: 2× 200 Gbps QSFP56 per node — one via switch (`enp1s0f1np1` / `rocep1s0f1`), one direct cable (`enP2p1s0f1np1` / `roceP2p1s0f1`)
- **NCCL**: Dual-HCA configured. In practice NCCL uses the switch link for the active TP all-reduce (bandwidth is not the bottleneck — only ~14 MB/s sustained — so the P2P cable sees minimal traffic).

## Software

- vLLM fork: `jasl/vllm` (DSV4 Flash support, MHC attention, MTP heads)
- Docker image: `vllm-dsv4-patched:latest` (custom patches applied — see `patches/`)
- NCCL 2.29.7, CUDA `sm_121a`, FP8 KV cache (UE8M0 scale format)
- Triton kernels cached on the `vllm-triton-cache` Docker volume

## Hardware Performance Ceiling

The throughput ceiling on this hardware is ≈30 tok/s with `spec=3`. Breakdown:

- Base decode (no spec): ≈12.8 tok/s, limited by memory bandwidth of HBM3e + LPDDR5X shared memory on GB10 (the 74 GB FP8 model spills out of the 20 GB HBM into the slower NVLink-C2C-attached LPDDR5X).
- MTP speculative speedup at observed acceptance: ≈2.35×, giving ~30 tok/s.
- Inter-node NCCL all-reduce (per layer, per step) adds <10% overhead — not the bottleneck.

This is consistent with the physics of the platform. Higher throughput would require a smaller-quantized model (FP4 would fit in HBM entirely) or higher-bandwidth interconnect to a unified KV cache pool.

## Known Behaviors

### `max_num_batched_tokens=32768` boundary

A prompt of **exactly** 32,768 tokens triggers vLLM's single-chunk fast path in chunked prefill. Combined with `spec=3` verification (batch=4) against 128 KV blocks, this combination deadlocks NCCL and locks both nodes for 30+ minutes. SSH/ping completely unreachable.

Verified across multiple power cycles. None of `VLLM_RINGBUFFER_WARNING_INTERVAL=3600`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`, or the `check_health` timeout patch prevent it.

**Avoid:** prompts of exactly 32,768 tokens. Use ≤ 32,512 or ≥ 33,024 (which forces chunked prefill across multiple chunks and works). For typical FreeLinus-style agentic workloads (≤16k context), this is a non-issue.

### Long-context performance

| Input tokens | TTFT |
|--------------|------|
| 8k | ~22s |
| 16k | ~28s |
| 28k | ~48s |
| 40k | ~57s |
| 60k | ~118s |

Prefill is roughly O(n²) due to attention against accumulated KV cache. Decode after prefill remains at ~30 tok/s regardless of context length.

### Prefix caching warm-up

With `--enable-prefix-caching`, repeated requests sharing a system prompt see TTFT drop from ~7s (cold) to ~2s (warm) on turn 2+ of a conversation. This is why FreeLinus tool-call loops feel responsive in practice — only the first turn of a session pays full prefill cost.

## Repository Layout

```
scripts/
  launch_head_server.sh      # Head node (spark-2)
  launch_worker_headless.sh  # Worker node (spark-1)
  full_warmup.py             # Post-boot Triton kernel + KV cache warmup
  abliterate.py              # SVD-based refusal direction projection (one-time, on weights)
patches/
  multiproc_executor_patch.py  # Two vLLM patches (applied via docker commit)
config/
  network.md                 # RoCE / NCCL / NFS topology
```

## Bring-up Procedure

After both nodes boot:

```bash
# 1. NFS export (on spark-2, head)
sudo exportfs -a

# 2. NFS mount (on spark-1, worker)
sudo mount 192.168.100.2:/home/sunil/models /home/sunil/models -t nfs -o vers=4,ro

# 3. Apply vLLM patches to the image (one time per image rebuild)
docker run --name vllm_patch_tmp -d vllm-dsv4-patched:latest sleep 300
docker cp patches/multiproc_executor_patch.py vllm_patch_tmp:/tmp/
docker exec vllm_patch_tmp python3 /tmp/multiproc_executor_patch.py
docker commit vllm_patch_tmp vllm-dsv4-patched:latest
docker rm -f vllm_patch_tmp

# 4. Start worker (spark-1) then head (spark-2)
ssh spark-1 'nohup bash /home/sunil/launch_worker_headless.sh > /tmp/vllm_worker.log 2>&1 &'
nohup bash /home/sunil/launch_head_server.sh > /tmp/vllm_head.log 2>&1 &

# 5. Wait for server (~5-8 min for model load + cudagraph capture)
until curl -sf http://localhost:8000/health; do sleep 15; done

# 6. Run warmup (compiles all Triton kernels, ~15-25 min)
python3 /tmp/full_warmup.py http://localhost:8000
```

After warmup, the server is ready for production traffic. The compiled Triton kernels are persisted to the `vllm-triton-cache` Docker volume, so subsequent cold boots only need warmup if the volume was wiped.

## vLLM Patches (`patches/multiproc_executor_patch.py`)

Two patches applied to `vllm/v1/executor/multiproc_executor.py` inside the container, then committed back to the image:

1. **Follower `collective_rpc` guard** — `assert self.rpc_broadcast_mq is not None` fires on the follower rank during init; guard returns `[]` for the follower so init succeeds.

2. **`check_health` timeout 10s → 3600s** — the default `check_health` RPC times out after 10s, but Triton JIT for the paged attention kernel at large KV cache sizes (e.g. batch=4 × 128 blocks) can block the GPU longer. The 10s timeout caused the executor to kill the worker mid-NCCL, hanging both nodes. The patch raises this to 3600s.

Both patches are idempotent and safe to re-run.

## Key Environment Variables

The launch scripts set these critical envs. Defaults don't work on this stack:

```bash
TORCH_CUDA_ARCH_LIST=12.1a               # SM_121a for GB10
VLLM_TRITON_MLA_SPARSE=1                 # Required; 0 falls back to a kernel not built for SM12x
NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1  # Both RoCE devices visible to NCCL
NCCL_IB_GID_INDEX=3                      # RoCEv2 IPv4-mapped GID
NCCL_CUMEM_ENABLE=0                      # Required for stability on GB10
VLLM_RINGBUFFER_WARNING_INTERVAL=3600    # shm_broadcast heartbeat — long enough for long-context prefill
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600  # Long prefills take time
DG_JIT_USE_NVRTC=0                       # Required; NVRTC path is broken for DSV4 MoE
DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
```

## Abliteration

`scripts/abliterate.py` performs single-pass SVD-based refusal-direction projection on the FP8 weight matrices. It produces `DeepSeek-V4-Flash-abliterated/` next to the original. The abliteration modifies both the main attention/MLP layers AND the MTP head weights (last shard differs from the original).

Empirically, the abliterated MTP heads give similar position-0 acceptance (≈84%) but slightly lower position-1/2 acceptance than the original MTP heads against the abliterated main model. Both configurations yield ≈2.3-2.5x spec speedup. Using the abliterated MTP heads (same model path in the speculative-config) is the simpler config because it doesn't require keeping both model copies on disk.

## License & Attribution

This deployment uses upstream weights and the `jasl/vllm` fork; respect the licenses of both. The abliteration script and orchestration here are MIT.
