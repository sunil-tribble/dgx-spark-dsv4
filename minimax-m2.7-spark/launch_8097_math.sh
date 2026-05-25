#!/bin/bash
# Math MiniMax-M2.7 instance: --reasoning auto, ctx 32K, NO --mlock (shares weights with 8096).
set -u
MODEL_DIR=/home/sunil/models/MiniMax-M2.7/UD-IQ4_XS
MODEL_FIRST="$MODEL_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
LLAMA_LIB=/home/sunil/swarm-llama
LLAMA_BIN="$LLAMA_LIB/llama-server"

LD_LIBRARY_PATH="$LLAMA_LIB" exec "$LLAMA_BIN" \
  -m "$MODEL_FIRST" \
  --alias minimax-m2.7-math \
  -ngl 999 \
  -c 32768 \
  -b 512 \
  -ub 128 \
  -fa on \
  -ctk q4_0 \
  -ctv q4_0 \
  --parallel 1 \
  --cache-ram 0 \
  --no-cache-prompt \
  --ctx-checkpoints 0 \
  --jinja \
  --reasoning auto \
  --reasoning-format deepseek \
  --reasoning-budget 4096 \
  --threads 8 \
  --host 0.0.0.0 \
  --port 8097
