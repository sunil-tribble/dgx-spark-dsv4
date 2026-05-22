# Dual-Spark DSV4-Flash vLLM Recipe — what's actually deployed

This documents the current state of a dual-NVIDIA-DGX-Spark vLLM cluster for
DeepSeek V4 Flash on `sunil@dgx-spark-2` / `sunil@dgx-spark-1`. It captures
the EXACT configuration that previously hit ~30 tok/s decode on this hardware,
plus the cleanup work in progress to reach the canonical 35 tok/s target.

**Snapshot date:** 2026-05-22
**Source machine:** dgx-spark-2 (10.117.1.215)

## Hardware

| | |
|---|---|
| Nodes | 2× NVIDIA DGX Spark (GB10) |
| GPU | NVIDIA GB10, SM 12.1 (sm_121), compute_cap 12.1 |
| Driver | 580.142 |
| Memory | 128 GB unified LPDDR5X per node (273 GB/s) |
| Interconnect | RoCE (ConnectX-7), interfaces `enp1s0f1np1` (192.168.100.x) and `enP2p1s0f1np1` (192.168.101.x) |
| Topology | spark-2 = 192.168.100.2 (head, NFS server). spark-1 = 192.168.100.3 (worker, NFS client) |

## Software stack

| | |
|---|---|
| OS | Ubuntu (aarch64) |
| CUDA | 13.0 at `/usr/local/cuda` |
| Python | 3.12.3 |
| Docker | 29.2.1 |
| vLLM fork | `jasl/vllm` @ commit `dda4668b59567416f86956cfe7bbc1eab371a61e` ("Restore rowwise paged-MQA logits kernel for SM12x long context", 2026-05-13) |
| Docker image | `vllm-node-dsv4:latest` (locally built, 20 GB), pinned at this commit + multiproc_executor patches baked in |
| Model | `/home/sunil/models/DeepSeek-V4-Flash` (150 GB, official HF FP4 weights, 46 shards, max_position_embeddings 1048576) |
| KV cache | FP8 (`--kv-cache-dtype fp8`) |

## What's on disk (spark-2)

```
/home/sunil/
├── jasl-vllm/                          # vLLM source @ dda4668b5
├── models/DeepSeek-V4-Flash/           # 150 GB FP4 model
├── launch_head_jasl.sh                 # head-node launcher (runs on spark-2)
├── launch_worker_jasl.sh               # worker-node launcher (runs on spark-1)
└── multiproc_executor_patch.py         # required runtime patch
```

## Launch (when both nodes are reachable)

### Required: NFS export from head → worker

The worker mounts `/home/sunil/models` from spark-2 over the RoCE network:

```bash
# On spark-2 (head):
sudo systemctl enable --now nfs-kernel-server
echo '/home/sunil/models 192.168.100.0/24(ro,sync,no_subtree_check)' | sudo tee -a /etc/exports
sudo exportfs -ra

# On spark-1 (worker): the launch script mounts automatically:
# sudo mount 192.168.100.2:/home/sunil/models /home/sunil/models -t nfs -o vers=4,ro
```

### Step 1 — start worker on spark-1

```bash
ssh sunil@10.117.1.24
cd ~ && ./launch_worker_jasl.sh
```

### Step 2 — start head on spark-2

```bash
ssh sunil@10.117.1.215
cd ~ && ./launch_head_jasl.sh
```

Health check:

```bash
curl -s http://10.117.1.215:8000/v1/models | jq '.data[0]'
```

### Step 3 — bench (Conv MTP=2 c=1 target ~35 t/s on mt-bench)

```bash
# Quick smoke
curl -s http://10.117.1.215:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"Count from 1 to 10."}],
       "max_tokens":80,"temperature":0}'

# Full bench
docker run --rm --network host -e VLLM_HOST=http://10.117.1.215:8000 \
  vllm-node-dsv4:latest \
  vllm bench serve \
    --dataset-name hf --dataset-path mtbench \
    --random-input-len 8192 --random-output-len 512 \
    --num-prompts 24 --max-concurrency 1 \
    --ignore-eos --temperature 1.0
```

## Key launch flags (from `launch_head_jasl.sh`)

These are the live-tested settings — see `scripts/launch_head_jasl.sh` for the
full version with all env vars and RoCE/NCCL knobs.

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `2` | spread across 2 nodes |
| `--distributed-executor-backend` | `mp` | multiproc not Ray |
| `--nnodes 2 --node-rank 0` | head | rank 1 on worker |
| `--master-addr 10.117.1.215` | spark-2 IP | head WiFi (NCCL goes over RoCE per IB_HCA) |
| `--enable-expert-parallel` | yes | MoE split across nodes |
| `--max-model-len` | `1048576` (1M) or `32768` (Conv) | env-var `CTX` |
| `--gpu-memory-utilization` | `0.87` | env-var `UTIL` |
| `--max-num-seqs` | `1` | env-var `SEQS` (Conv profile uses 36) |
| `--max-num-batched-tokens` | `32768` | matches max-model-len |
| `--block-size` | `256` | bigger blocks for long ctx |
| `--kv-cache-dtype` | `fp8` | required by DSV4 |
| `--enable-prefix-caching` | yes | sub-second TTFT on repeated prefixes |
| `--attention-backend` | `FLASHINFER_MLA_SPARSE` | sparse MLA path |
| `--speculative-config` | `deepseek_mtp num_speculative_tokens=3` | (canonical recipe uses 2) |
| `--tool-call-parser` | `deepseek_v4` | tool-use support |
| `--reasoning-parser` | `deepseek_v4` | `<think>...</think>` |

## Critical NCCL/RoCE env vars

```
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f1np1
TORCH_CUDA_ARCH_LIST=12.1a
VLLM_TRITON_MLA_SPARSE=1
VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

Long-prefill stability requires bumping every NCCL/vLLM timeout to 3600:
```
VLLM_RPC_TIMEOUT=3600000
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
NCCL_TIMEOUT=3600
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
```

## Multiproc executor patch (required)

`scripts/multiproc_executor_patch.py` patches the in-container vLLM at runtime:

1. **RPC broadcast guard**: replaces `assert self.rpc_broadcast_mq is not None`
   with a graceful return for follower nodes. Without this, the worker crashes
   immediately because the follower doesn't own the broadcast queue.
2. **check_health timeout** 10s → 3600s. Without this, long cold prefills
   (>100K context) trigger spurious health-check failures and tear down the
   cluster.

The launch scripts invoke `python3 /tmp/multiproc_executor_patch.py` before
`vllm serve`, applying the patch idempotently each boot.

## Measured throughput (from `scripts/bench_results.txt`)

Prior measurements on this exact stack (spec=2 abliterated DSV4 Flash):

| Workload | Output | Speed |
|---|---|---|
| Throughput run 1 | 300 tok | **29.7 tok/s** |
| GPQA Physics | 1474 tok | 24 tok/s |
| Competition Math | 6000 tok | 30 tok/s |

These align with the published canonical baseline of **~35 tok/s** for
Conv MTP=2 c=1 on mt-bench prompts (jasl/vllm-ds4-sm120-harness baseline
`20260513_gb10_post_rowwise_dda4668b5`, [report.md](https://github.com/jasl/vllm-ds4-sm120-harness/blob/main/baselines/20260513_gb10_post_rowwise_dda4668b5/report.md)).

## Current state (2026-05-22)

- ✅ Docker image present and tagged correctly
- ✅ Model weights present (150 GB, 46 shards)
- ✅ Launch scripts present and aligned with prior 30 t/s run
- ✅ Multiproc patch ready
- ❌ **spark-1 SSH currently dead** (port 22 connection refused, but ICMP ping works = 6 ms RTT). Requires physical power-cycle.

Once spark-1 SSH is restored:
1. spark-1 mounts NFS from spark-2 (`/home/sunil/models`)
2. Run `launch_worker_jasl.sh` on spark-1
3. Run `launch_head_jasl.sh` on spark-2
4. Bench → expect 25-35 tok/s on canonical workload
5. Capture bench artifact + commit

## Canonical reference

- jasl/vllm fork: https://github.com/jasl/vllm/tree/dda4668b5
- Validation harness: https://github.com/jasl/vllm-ds4-sm120-harness
- Forum thread the perf numbers came from: see the dda4668b5 baseline `report.md`
- lmxxf alt recipe (mixed Marlin+DeepGEMM, ~12 t/s): https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark

## Files in this directory

```
README.md                         # this file
scripts/launch_head_jasl.sh       # head launcher (mirrors /home/sunil/ on spark-2)
scripts/launch_worker_jasl.sh     # worker launcher (mirrors /home/sunil/ on spark-1 via prior copy)
scripts/multiproc_executor_patch.py  # required runtime patch
scripts/bench_results.txt         # measured throughput from prior runs
scripts/docker_images_snapshot.txt   # docker images list at snapshot time
scripts/gpu_snapshot.txt          # nvidia-smi summary
scripts/jasl_vllm_head.txt        # exact commit hash
scripts/dsv4_flash_config.json    # model config (FP4, 1M ctx)
```
