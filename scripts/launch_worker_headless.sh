#!/bin/bash
# Worker — matches head config: spec=3, abliterated MTP, 3600s watchdog
set -euo pipefail
MODEL=/home/sunil/models/DeepSeek-V4-Flash-abliterated
MODEL_ORIG=/home/sunil/models/DeepSeek-V4-Flash
IMAGE=vllm-dsv4-patched:latest
CONTAINER=vllm_ds4_worker

# Ensure NFS mount is live
if ! mountpoint -q /home/sunil/models; then
  sudo mount 192.168.100.2:/home/sunil/models /home/sunil/models -t nfs -o vers=4,ro
fi

docker rm -f "$CONTAINER" 2>/dev/null || true
exec docker run --name "$CONTAINER" \
  --privileged --ipc=host --net=host --gpus all --shm-size 64g \
  -v "$MODEL":/models/DeepSeek-V4-Flash:ro \
  -v "$MODEL_ORIG":/models/DeepSeek-V4-Flash-orig:ro \
  -v /home/sunil/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  -v vllm-triton-cache:/root/.triton \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e DG_JIT_USE_NVRTC=0 \
  -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  -e NCCL_IB_DISABLE=0 \
  '-e' 'NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1' \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  -e VLLM_RINGBUFFER_WARNING_INTERVAL=3600 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e VLLM_WORKER_PROC_LAUNCH_TIMEOUT_S=3600 \
  -e VLLM_RPC_TIMEOUT=3600000 \
  -e TRITON_CACHE_INVALIDATE=0 \
  -e HF_DATASETS_OFFLINE=1 \
  "$IMAGE" \
  vllm serve /models/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-prefix-caching \
    --max-model-len 1048576 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.90 \
    --no-enable-flashinfer-autotune \
    '--compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    '--speculative-config={"method":"deepseek_mtp","num_speculative_tokens":3,"model":"/models/DeepSeek-V4-Flash"}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    '--reasoning-config={"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
    '--default-chat-template-kwargs={"thinking":true}' \
    --load-format safetensors \
    --headless \
    --nnodes 2 --node-rank 1 \
    --master-addr 10.117.1.215 --master-port 29501
