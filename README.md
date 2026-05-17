# 2x DGX GB10 Spark — DeepSeek V4 Flash: Fast, Jailbroken, 1M Context

## Overview

This repo documents a fully operational DeepSeek V4 Flash deployment across two NVIDIA DGX GB10 Spark nodes. What you get:

- **1M token context** (1,163,321 KV cache tokens available at `gpu_memory_utilization=0.87`)
- **28–35 tok/s** single-stream decode
- **MTP speculative decoding** (`deepseek_mtp`, 2 draft tokens, ~60% acceptance rate)
- **Expert parallel** (`EP=2`, 256 experts split evenly across both nodes — 128 each)
- **Dual-rail RoCE v2** (2×200Gbps QSFP56 direct cable, bonded for NCCL all-to-all)
- **Abliterated weights** (refusal direction projected out via SVD; full compliance removal)
- **32K batched tokens** (supports full long-form reasoning chains without truncation)

The deployment uses a custom vLLM fork with DeepSeek V4 Flash support, FP8 KV cache, MTP drafting, and multi-node MP (not Ray) for correct single-GPU-per-node topology.

---

## Hardware

| Node | Role | Memory | NICs |
|------|------|--------|------|
| spark-2 | HEAD (vLLM server) | 121.69 GiB unified (CPU+GPU shared) | 2× 200Gbps QSFP56 |
| spark-1 | WORKER (headless follower) | 121.69 GiB unified (CPU+GPU shared) | 2× 200Gbps QSFP56 |

- GB10 Grace Blackwell Superchip: 1× B200 GPU + 72-core Grace CPU, NVLink-C2C connected (unified memory, no PCIe copies)
- Direct cable between nodes: `enp1s0f1np1` (primary) and `enP2p1s0f1np1` (secondary), both 200Gbps
- RoCE v2 ACTIVE on both NICs, GID index 3

See `config/network.md` for full topology and NFS configuration.

---

## Software Stack

| Component | Value |
|-----------|-------|
| vLLM fork | `jasl/vllm` @ `dda4668b59567416f86956cfe7bbc1eab371a61e` |
| Docker image | `vllm-dsv4-jasl:latest` |
| Base image | `lmxxf/vllm-deepseek-v4-dgx-spark:latest` + jasl Python overlay |
| Model | `deepseek-ai/DeepSeek-V4-Flash` FP8 (abliterated variant) |
| Python | 3.11 (inside container) |
| CUDA | 12.8 |

---

## Key Technical Discoveries

These are non-obvious problems encountered during bring-up. Each one cost significant debugging time.

### 1. Gloo Hostname Fix (critical for multi-node)

spark-2's `/etc/hosts` has `127.0.1.1 dgx-spark-2`. When the vLLM head node initializes the distributed process group using Gloo (used for MP coordinator), it resolves the hostname to `127.0.1.1` and binds there. The worker on spark-1 can never reach `127.0.1.1`. Rendezvous hangs indefinitely.

Fix applied in `launch_head.sh`:
```
--add-host dgx-spark-2:10.117.1.215   # Docker flag — overrides inside container
--master-addr 10.117.1.215            # vLLM flag — explicit bind address
```

Error symptom (without fix): worker logs `[W ProcessGroupGloo.cpp] Gloo connectFullMesh failed with ...` or the worker simply hangs at "Waiting for all workers".

### 2. Distributed Backend: MP not Ray

Each DGX GB10 Spark has one GPU. Ray's topology detection assumes `local_world_size` equals visible devices per node. With 2 nodes × 1 GPU and `tensor_parallel_size=2`, Ray miscalculates the local world size and raises:

```
AssertionError: local_world_size (2) > num visible devices (1)
```

Use `--distributed-executor-backend mp` instead. The worker is launched with `--headless` and `--node-rank 1`; the head serves the API.

### 3. RoCE v2 for NCCL (expert-parallel all-to-all)

Default NCCL over TCP gives ~40 GB/s effective bandwidth between nodes. With RoCE v2 over the 200Gbps direct link, you get ~140 GB/s. This matters enormously for expert-parallel: 256 experts across 2 nodes means every MoE layer dispatches tokens across the link.

Required env vars:
```
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_CUMEM_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1
```

`NCCL_CUMEM_ENABLE=0`: Grace Blackwell unified memory conflicts with NCCL's CUDA memory pool — disable it.

Verify RoCE is active before relying on it:
```bash
show_gids | grep RoCE
# Should show GID index 3 as type RoCE v2 on both rocep1s0f1 and roceP2p1s0f1
```

### 4. VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

At large context (917K+), vLLM's default memory profiler pre-estimates CUDA graph memory by running dummy forward passes at every sequence length up to `max_model_len`. At 1M context this takes 15–20 minutes and can OOM.

Setting `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` skips this estimation. CUDA graphs still build correctly on first use; only the pre-estimation is skipped. Startup time drops from 15–20 minutes to ~8 minutes (including JIT compilation on first run, ~3 minutes on subsequent runs with warm cache).

### 5. TRANSFORMERS_OFFLINE=1 (required for abliterated weights)

When serving a local directory that has been modified (abliterated safetensors), transformers attempts to validate the local path as a HuggingFace Hub repo ID. This triggers an outbound HTTPS request. If the path doesn't resolve as a valid Hub ID, you get confusing auth errors or silent hangs at startup.

```
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

These must be set in both head and worker containers when serving the abliterated model.

### 6. NFS for Model Distribution

spark-1 has no independent copy of the model. spark-2 exports `/home/sunil/models` over the 192.168.100.0/24 direct link (200Gbps). spark-1 mounts it read-only at the same path.

NFS read performance over this link is ~12 GB/s, well above the ~3 GB/s needed to load 685B FP8 parameters in a reasonable time. Model loading on the worker takes the same time as on the head.

### 7. Expert Parallel

`--enable-expert-parallel` routes MoE expert computation across both nodes. Without this flag, both GPUs hold the full set of 256 experts and the cross-node traffic is limited to tensor-parallel all-reduce. With EP=2, expert routing becomes a true all-to-all across the link, but each GPU only needs to compute 128 experts, roughly halving per-expert compute.

At 28–35 tok/s decode with EP=2, removing EP drops throughput to ~18 tok/s (bottleneck shifts to expert compute on a single GPU).

### 8. Context Scaling

With `gpu_memory_utilization=0.87`, FP8 KV cache, and `block_size=256`:

- Available KV cache tokens: **1,163,321**
- `max_model_len=1048576` (1M context) leaves ~115K tokens headroom
- Actual 1M context prompt processing is feasible with `max_num_seqs=1`

Increasing utilization to 0.90 recovers ~60K more KV tokens but risks OOM during JIT compilation. 0.87 is the stable ceiling.

### 9. MTP Drafter Config Lookup

When `--speculative-config` uses `method: deepseek_mtp` and the served model is an abliterated local directory, vLLM's drafter config loader fails to find the MTP head configuration because the abliterated model dir may lack the original `config.json` drafter spec entries.

Fix: explicitly pass the original model path as the drafter config source:
```json
{"method":"deepseek_mtp","num_speculative_tokens":2,"model":"/models/DeepSeek-V4-Flash-orig"}
```

The actual drafter **weights** still come from the main model checkpoint (the MTP head is shared and present in the abliterated shards). Only the config lookup uses the original path. Mount the original model read-only as a second volume:
```
-v /home/sunil/models/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash-orig:ro
```

### 10. Compile Cache Persistence

First launch builds:
- DeepGroup JIT kernels (DG_JIT via NVRTC): ~15 minutes
- vLLM CUDA graph capture: ~5 minutes

Mount a named Docker volume to persist this cache:
```
-v vllm-cache:/root/.cache/vllm
```

Subsequent restarts skip JIT compilation and load from cache: total startup ~3 minutes after warm cache. Without this volume, every restart pays the full 20-minute compilation cost.

---

## Abliteration Approach

The refusal suppression mechanism in DeepSeek V4 Flash is encoded in the shared expert weights across early layers. The approach removes it via orthogonal projection:

1. **Load gate weights** (`ffn.shared_experts.w1.weight`) from layers 0–8
2. **SVD** of each weight matrix; take the top right singular vector (principal direction in input space)
3. **Average** across layers to get a robust, layer-invariant refusal direction
4. **Project out** from `w1` (gate), `w3` (up proj), `w2` (down proj) for all shared experts, and from `attn.wo_b.weight` (attention output bias)
5. **FP8 round-trip**: cast `float8_e4m3fn → float32` for projection; clamp to `[-448, 448]` (FP8 max); cast back. Scale tensors (`e8m0fnu`) are left unchanged — only the quantized weight values are modified.

The projection formula for input-space direction `d`:
```
W_new = W - alpha * (W @ d).unsqueeze(1) * d.unsqueeze(0)
```

`alpha=1.0` gives full removal. Partial removal (`alpha < 1.0`) can preserve some safety signal while reducing refusal rate, but 1.0 is used here for complete ablation.

See `scripts/abliterate.py` for the full implementation. Runtime on this hardware with NVMe-backed model storage: ~45 minutes per abliteration pass (dominated by FP8 decode/re-encode of 685B parameters split across shards).

---

## Performance Numbers

| Config | max_model_len | max_num_seqs | gpu_memory_utilization | tok/s (decode) | KV tokens |
|--------|---------------|--------------|------------------------|----------------|-----------|
| 200K baseline | 200K | 2 | 0.85 | 34 tok/s | ~812K |
| 917K single-stream | 917K | 1 | 0.87 | 28 tok/s | 1.04M |
| 1M single-stream | 1M | 1 | 0.87 | ~27 tok/s | 1.16M |

MTP acceptance rate: ~60% at typical prompt distributions. Effective throughput with MTP is approximately 1.6× naive autoregressive at the decode stage.

---

## Quick Start

### Prerequisites

- Both nodes running Ubuntu 22.04 with Docker and NVIDIA container toolkit
- `vllm-dsv4-jasl:latest` image present on both nodes (see build section below)
- Model downloaded to spark-2 at `/home/sunil/models/DeepSeek-V4-Flash`
- Abliterated model generated at `/home/sunil/models/DeepSeek-V4-Flash-abliterated`
- NFS export active on spark-2, NFS mount active on spark-1
- RoCE v2 verified on both nodes

### Step 1: Download the model (spark-2)

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /home/sunil/models/DeepSeek-V4-Flash \
  --include "*.safetensors" "*.json" "*.py" "*.tiktoken"
```

Approximately 685 GB. Use `--resume-download` if interrupted.

### Step 2: Build or pull the Docker image

The base image with DGX GB10 Spark support:
```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

Apply the jasl vLLM overlay (installs `jasl/vllm` @ `dda4668b`):
```bash
# On both nodes
docker build -t vllm-dsv4-jasl:latest - <<'EOF'
FROM lmxxf/vllm-deepseek-v4-dgx-spark:latest
RUN pip install --no-deps \
  "https://github.com/jasl/vllm/archive/dda4668b59567416f86956cfe7bbc1eab371a61e.tar.gz"
EOF
```

Or `docker save` from spark-2 and `docker load` on spark-1 to avoid redundant builds.

### Step 3: Abliterate the weights (spark-2)

```bash
cd /home/sunil/dgx-spark-dsv4
pip install safetensors torch  # if running outside container
python scripts/abliterate.py
```

Output at `/home/sunil/models/DeepSeek-V4-Flash-abliterated`. Takes ~45 minutes.

### Step 4: Configure NFS (spark-2 → spark-1)

On spark-2:
```bash
# Install NFS server
apt-get install -y nfs-kernel-server

# Add export
echo '/home/sunil/models  192.168.100.1(ro,no_root_squash,async,no_subtree_check)' \
  >> /etc/exports
exportfs -ra
systemctl restart nfs-kernel-server
```

On spark-1:
```bash
# Install NFS client
apt-get install -y nfs-common

mkdir -p /home/sunil/models
mount -t nfs -o ro,hard,intr,rsize=1048576 \
  192.168.100.2:/home/sunil/models /home/sunil/models

# Add to /etc/fstab for persistence
echo '192.168.100.2:/home/sunil/models /home/sunil/models nfs ro,hard,intr,rsize=1048576,wsize=1048576 0 0' \
  >> /etc/fstab
```

### Step 5: Verify RoCE

On both nodes:
```bash
show_gids | grep -E 'RoCE|GID'
# Confirm GID index 3 is RoCE v2 on rocep1s0f1 and roceP2p1s0f1

# Connectivity test
ibping -S  # on spark-1
ibping -c 10 -L 1 192.168.100.2  # on spark-2
```

### Step 6: Launch head (spark-2)

```bash
chmod +x /home/sunil/dgx-spark-dsv4/scripts/launch_head.sh
/home/sunil/dgx-spark-dsv4/scripts/launch_head.sh
```

Wait for: `Waiting for 1 worker nodes` in the container logs.

### Step 7: Launch worker (spark-1)

Copy `scripts/launch_worker.sh` to spark-1 (or scp from spark-2), then:

```bash
chmod +x launch_worker.sh
./launch_worker.sh
```

### Step 8: Wait for full startup

On spark-2, tail the head container logs:
```bash
docker logs -f vllm_ds4_head
```

Expected sequence:
1. `Loading model weights...` (3–5 min with warm NFS cache)
2. `Starting to profile...` + memory profiling (~2 min)
3. JIT compilation messages from DG_JIT (~3 min with warm vllm-cache volume)
4. `CUDA graph capture...` (~2 min)
5. `Uvicorn running on http://0.0.0.0:8000`

Total: ~8 minutes with warm caches. ~25 minutes cold.

### Step 9: Test

```bash
curl http://10.117.1.215:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100,
    "stream": false
  }'
```

---

## Troubleshooting

### Worker hangs at "Waiting for head"

**Cause**: Gloo hostname resolution binding to `127.0.1.1` inside the head container.  
**Fix**: Confirm `--add-host dgx-spark-2:10.117.1.215` is in `launch_head.sh`. Verify with `docker inspect vllm_ds4_head | grep HostAliases`.

### `AssertionError: local_world_size (2) > num visible devices (1)`

**Cause**: Using Ray backend instead of MP.  
**Fix**: Ensure `--distributed-executor-backend mp` is present (it's the default in launch scripts — check you're not passing `--distributed-executor-backend ray`).

### Startup takes 20+ minutes

**Cause**: `vllm-cache` Docker volume not warm (JIT recompilation on every start).  
**Fix**: Confirm the volume exists: `docker volume inspect vllm-cache`. On first run this is expected. Subsequent runs should be ~8 min.

### `OSError: We couldn't connect to 'https://huggingface.co'` or similar at startup

**Cause**: `TRANSFORMERS_OFFLINE` not set when serving abliterated (locally modified) weights.  
**Fix**: Confirm both `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are in the launch script env.

### NCCL falls back to socket transport

**Symptom**: `NCCL INFO Using network Socket` instead of `NCCL INFO Using network IB` in logs.  
**Cause**: RoCE GID index mismatch, or HCA name incorrect.  
**Fix**: Run `show_gids` and confirm index 3 is RoCE v2. Check `NCCL_IB_HCA` matches output of `ibv_devinfo` device names.

### MTP drafter config error: `KeyError` or `FileNotFoundError` in drafter init

**Cause**: Abliterated model dir missing original drafter config entries.  
**Fix**: Ensure `-v /home/sunil/models/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash-orig:ro` is mounted and `"model":"/models/DeepSeek-V4-Flash-orig"` is in the speculative config JSON.

### OOM during startup

**Cause**: `gpu_memory_utilization` set above 0.87, or `NCCL_CUMEM_ENABLE=1`.  
**Fix**: Use `gpu_memory_utilization=0.87`. Confirm `NCCL_CUMEM_ENABLE=0` — Grace Blackwell unified memory is incompatible with NCCL's CUDA memory pool allocator.

### `ibv_reg_mr failed` or NCCL registration errors

**Cause**: `--ipc=host` missing from docker run, or container not `--privileged`.  
**Fix**: Both `--privileged` and `--ipc=host` are required for RDMA memory registration.

---

## Files

```
dgx-spark-dsv4/
├── README.md                  # This file
├── scripts/
│   ├── launch_head.sh         # Launch vLLM server on spark-2 (head)
│   ├── launch_worker.sh       # Launch vLLM follower on spark-1 (worker)
│   └── abliterate.py          # FP8-aware abliteration script
└── config/
    └── network.md             # Network topology, RoCE config, NFS setup
```

## Recovery After Power Cycle (2026-05-17)

After power cycling both DGX Sparks, the following steps are needed:

### 1. Remount NFS on spark-1
```bash
sudo mount 192.168.100.2:/home/sunil/models /home/sunil/models -t nfs -o vers=4,ro
ls /home/sunil/models/DeepSeek-V4-Flash/config.json  # verify
```

### 2. Start worker on spark-1
```bash
bash /home/sunil/launch_worker_headless.sh
```

### 3. Start head on spark-2
```bash
nohup bash /home/sunil/launch_head_server.sh > /tmp/vllm_head.log 2>&1 &
tail -f /tmp/vllm_head.log  # watch progress
```

### 4. After ~10 min, verify:
```bash
curl http://localhost:8000/health
```

### 5. Update Hermes to restore FreeLinus to spark-2:
```bash
# On raoDesktop-wsl:
cp /home/sunil/.hermes/config.yaml.bak_spark2_down /tmp/bak.yaml
# Change base_url back to http://100.117.56.97:8000/v1 in config.yaml
systemctl --user restart hermes-gateway.service
```

### Image: vllm-dsv4-jasl:latest
Same as original jasl fork (`dda4668b5`), locally cached as `vllm-dsv4-jasl:latest`.
Key env vars: TORCH_CUDA_ARCH_LIST=12.1a, VLLM_TRITON_MLA_SPARSE=1, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
