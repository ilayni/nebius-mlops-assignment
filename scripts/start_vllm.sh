#!/usr/bin/env bash
# Start vLLM serving Qwen3-30B-A3B-Instruct on a single H100.
# First-time startup: ~35 min (triton kernel compilation + weight load).
# Subsequent restarts: ~5-10 min (kernels cached in ~/.triton).
# DO NOT restart between phases — each restart burns 5-35 min of H100 time.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3-30b}"
PORT="${VLLM_PORT:-8000}"
LOG="${VLLM_LOG:-$HOME/vllm.log}"

# Download weights if not already present (~2.5 min, ~60 GB).
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Downloading weights to $MODEL_DIR ..."
  huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct --local-dir "$MODEL_DIR"
fi

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  \
  `# --dtype bfloat16` \
  `#   Qwen3-30B ships in bfloat16; avoids a re-quantisation pass on H100 NVLink.` \
  --dtype bfloat16 \
  \
  `# --tensor-parallel-size 1` \
  `#   Single H100 — no inter-GPU split needed; TP>1 adds NCCL overhead for nothing.` \
  --tensor-parallel-size 1 \
  \
  `# --gpu-memory-utilization 0.90` \
  `#   Leaves ~8 GB headroom for KV-cache growth under load bursts; 0.95+ risks OOM.` \
  --gpu-memory-utilization 0.90 \
  \
  `# --max-model-len 8192` \
  `#   SQL queries + schema fit comfortably in 8 k tokens; shorter window = larger KV cache.` \
  --max-model-len 8192 \
  \
  `# --enable-chunked-prefill` \
  `#   Splits long prefills across scheduler steps so decode requests aren't starved;` \
  `#   keeps p95 latency stable under mixed short/long prompt workloads.` \
  --enable-chunked-prefill \
  \
  --port "$PORT" > "$LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM starting (PID=$VLLM_PID) — warmup takes ~35 min first time."
echo "Logs: tail -f $LOG"
echo "Health check: until curl -sf http://localhost:$PORT/v1/models; do sleep 10; done"
