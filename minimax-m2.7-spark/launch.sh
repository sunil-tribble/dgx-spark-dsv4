#!/usr/bin/env bash
# Launch MiniMax-M2.7-UD-IQ4_XS on a single DGX Spark via llama.cpp.
#
# Sizing on Spark (128 GB nominal / 121 GB user-accessible after kernel reserve):
#   weights (UD-IQ4_XS) ............. 108.4 GB
#   KV cache (q8_0, 32K ctx) ........  ~6  GB
#   activations + cuda graphs ........  ~3  GB
#   total ........................... ~117 GB  -> fits with ~4 GB headroom
#
# MiniMax-M2 is MoE (230B-A10B): 10B active per token across the unified-memory
# bandwidth, so decode is bandwidth-limited like DSV4. Expected ~15-25 t/s decode
# on Spark depending on prompt patterns and KV cache hit rate.

set -euo pipefail

MODEL_DIR=/home/sunil/models/MiniMax-M2.7/UD-IQ4_XS
MODEL_FIRST="$MODEL_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
LLAMA_LIB=/home/sunil/swarm-llama
LLAMA_BIN="$LLAMA_LIB/llama-server"

# Sanity
[ -f "$MODEL_FIRST" ] || { echo "missing first shard: $MODEL_FIRST"; exit 1; }
for i in 1 2 3 4; do
  f="$MODEL_DIR/MiniMax-M2.7-UD-IQ4_XS-0000$i-of-00004.gguf"
  [ -f "$f" ] || { echo "missing shard $i: $f"; exit 1; }
done

CTX=${CTX:-32768}      # 32K context — leaves headroom for KV
NGL=${NGL:-999}        # offload everything
PORT=${PORT:-8096}     # next free port (5000/8082/8083/8094/8095 taken)
ALIAS=${ALIAS:-minimax-m2.7}
THREADS=${THREADS:-12}
BATCH=${BATCH:-512}
UBATCH=${UBATCH:-256}

# llama.cpp loads all shards automatically when given the first one
LD_LIBRARY_PATH="$LLAMA_LIB" exec "$LLAMA_BIN" \
  -m "$MODEL_FIRST" \
  --alias "$ALIAS" \
  -ngl "$NGL" \
  -c "$CTX" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --parallel 1 \
  --cache-ram 0 \
  --no-cache-prompt \
  --ctx-checkpoints 0 \
  --jinja \
  --reasoning auto \
  --reasoning-format deepseek \
  --reasoning-budget 4096 \
  --threads "$THREADS" \
  --mlock \
  --host 0.0.0.0 \
  --port "$PORT"
