#!/bin/bash
# MiniMax-M2.7 with reasoning disabled (v3 experiment).
set -u
MODEL_DIR=/home/sunil/models/MiniMax-M2.7/UD-IQ4_XS
MODEL_FIRST="$MODEL_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
LLAMA_LIB=/home/sunil/swarm-llama
LLAMA_BIN="$LLAMA_LIB/llama-server"

CTX=${CTX:-196608}
NGL=${NGL:-999}
PORT=${PORT:-8096}
ALIAS=${ALIAS:-minimax-m2.7}
THREADS=${THREADS:-12}
BATCH=${BATCH:-512}
UBATCH=${UBATCH:-256}
KTYPE=${KTYPE:-q4_0}
VTYPE=${VTYPE:-q4_0}

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
  --reasoning off \
  --reasoning-budget 0 \
  --threads "$THREADS" \
  --mlock \
  --host 0.0.0.0 \
  --port "$PORT"
