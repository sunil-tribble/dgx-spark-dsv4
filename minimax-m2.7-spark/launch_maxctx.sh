#!/bin/bash
# MiniMax-M2.7 UD-IQ4_XS — push to max context (192K) via q4_0 KV cache.
#
# Budget at 192K:
#   weights ........ 108 GB (fixed)
#   K cache (q4_0) ..  6.2 GB
#   V cache (q4_0) ..  6.2 GB
#   compute buf ....  ~1 GB
#   total .......... ~121 GB  -> at the wall, expect to live
#
# If OOM: drop to CTX=131072 (128K) or use q8_0 K + q4_0 V (asymmetric).

set -u
MODEL_DIR=/home/sunil/models/MiniMax-M2.7/UD-IQ4_XS
MODEL_FIRST="$MODEL_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
LLAMA_LIB=/home/sunil/swarm-llama
LLAMA_BIN="$LLAMA_LIB/llama-server"

CTX=${CTX:-196608}        # MiniMax-M2.7's training-time max
NGL=${NGL:-999}
PORT=${PORT:-8096}
ALIAS=${ALIAS:-minimax-m2.7}
THREADS=${THREADS:-12}
BATCH=${BATCH:-512}
UBATCH=${UBATCH:-256}
KTYPE=${KTYPE:-q4_0}      # K cache quant
VTYPE=${VTYPE:-q4_0}      # V cache quant

LD_LIBRARY_PATH="$LLAMA_LIB" exec "$LLAMA_BIN" \
  -m "$MODEL_FIRST" \
  --alias "$ALIAS" \
  -ngl "$NGL" \
  -c "$CTX" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  -fa on \
  -ctk "$KTYPE" \
  -ctv "$VTYPE" \
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
