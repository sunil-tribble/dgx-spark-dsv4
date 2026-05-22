#!/bin/bash
set -e
MODEL=/home/sunil/models/DeepSeek-V4-Flash-abliterated-v3
MODEL_ORIG=/home/sunil/models/DeepSeek-V4-Flash
CTX=${CTX:-1048576}
SEQS=${SEQS:-1}
UTIL=${UTIL:-0.87}

if ! mountpoint -q /home/sunil/models; then
  sudo mount 192.168.100.2:/home/sunil/models /home/sunil/models -t nfs -o vers=4,ro || true
fi

exec docker run --rm \
  --runtime nvidia --gpus all --network host --shm-size 64g \
  --entrypoint /bin/bash \
  -v ${MODEL}:/models/DeepSeek-V4-Flash:ro \
  -v ${MODEL_ORIG}:/models/DeepSeek-V4-Flash-orig:ro \
  -v /home/sunil/multiproc_executor_patch.py:/tmp/multiproc_executor_patch.py:ro \
  -v vllm-cache:/root/.cache \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_RINGBUFFER_WARNING_INTERVAL=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e VLLM_RPC_TIMEOUT=3600000 \
  -e VLLM_WORKER_PROC_LAUNCH_TIMEOUT_S=3600 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_TIMEOUT=3600 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
  -e TORCH_NCCL_BLOCKING_WAIT=0 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  -e DG_JIT_USE_NVRTC=0 \
  -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e TRITON_CACHE_INVALIDATE=0 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  vllm-node-dsv4:latest \
  -c 'python3 /tmp/multiproc_executor_patch.py && exec vllm serve /models/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4-flash \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank 1 \
    --master-addr 10.117.1.215 --master-port 29501 \
    --enable-expert-parallel \
    --headless \
    --max-model-len '${CTX}' \
    --gpu-memory-utilization '${UTIL}' \
    --max-num-seqs '${SEQS}' \
    --max-num-batched-tokens 8192 \
    --block-size 256 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --attention-backend FLASHINFER_MLA_SPARSE \
    --speculative-config '\''{"method":"deepseek_mtp","num_speculative_tokens":2,"model":"/models/DeepSeek-V4-Flash"}'\'''
