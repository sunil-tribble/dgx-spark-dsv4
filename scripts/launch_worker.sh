#!/bin/bash
# Worker node: spark-1 (192.168.100.1)
# Run this script ON SPARK-1 after the head node is up.
# The abliterated model is accessed via NFS mount from spark-2.
MODEL=/home/sunil/models/DeepSeek-V4-Flash-abliterated
IMAGE=vllm-dsv4-jasl:latest
CONTAINER=vllm_ds4_worker
MASTER_ADDR=10.117.1.215
CTX=1048576
SEQS=1
UTIL=0.87

docker rm -f $CONTAINER 2>/dev/null || true

docker run -d --name $CONTAINER \
  --privileged --ipc=host --net=host --gpus all --shm-size 64g \
  -v $MODEL:/models/DeepSeek-V4-Flash:ro \
  -v /home/sunil/.cache/huggingface:/root/.cache/huggingface \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e DG_JIT_USE_NVRTC=0 \
  -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  -e NCCL_IB_DISABLE=0 \
  -e 'NCCL_IB_HCA=rocep1s0f1:1,roceP2p1s0f1:1' \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  $IMAGE \
  vllm serve /models/DeepSeek-V4-Flash \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-prefix-caching \
    --max-model-len $CTX \
    --max-num-seqs $SEQS \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization $UTIL \
    --no-enable-flashinfer-autotune \
    '--compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    '--speculative-config={"method":"deepseek_mtp","num_speculative_tokens":2}' \
    --tokenizer-mode deepseek_v4 \
    --load-format safetensors \
    --nnodes 2 --node-rank 1 \
    --master-addr $MASTER_ADDR --master-port 29501 \
    --headless

echo "Worker started."
