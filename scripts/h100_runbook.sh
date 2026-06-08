#!/usr/bin/env bash
# H100 session runbook — execute top to bottom; do NOT skip steps or reorder.
# vLLM first-time warmup + kernel compilation takes ~35 min (cached after).
# DO NOT restart vLLM between phases — each restart costs ~35 min cold or ~5-10 min warm.
set -euo pipefail

H100_IP="${H100_IP:-FILL_ME_IN}"
REPO="${REPO:-git@github.com:YOUR_USER/mlops-assignment-private.git}"
MODEL_DIR=~/models/qwen3-30b

# ─── 0:00-0:05  SSH + clone + start vLLM ────────────────────────────────────
# Run this SSH command locally first (in a separate terminal):
#   ssh -L 3000:localhost:3000 -L 9090:localhost:9090 \
#       -L 3001:localhost:3001 -L 8000:localhost:8000 -L 8001:localhost:8001 \
#       user@$H100_IP

# On H100:
git clone "$REPO" app && cd app

# Download weights (~2.5 min), then start vLLM — warmup clock starts now
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct --local-dir "$MODEL_DIR"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000 > ~/vllm.log 2>&1 &

echo "vLLM starting — warmup takes ~35 min first time. PID=$!"
echo "Tail: tail -f ~/vllm.log"

# ─── 0:05-0:40  Install tooling + start o11y stack (vLLM warms up in background)
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y docker.io docker-compose python3-dev
source ~/.bashrc  # pick up uv

pip install -r requirements.txt

cp .env.example .env
# Edit .env: set OPENAI_BASE_URL=http://localhost:8000/v1, OPENAI_API_KEY=EMPTY
# (also fill in Langfuse keys after stack is up)
${EDITOR:-nano} .env

docker compose up -d

echo "Grafana:    http://localhost:3000   (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Langfuse:   http://localhost:3001"

# ─── 0:40-1:00  Validate full stack (vLLM should be ready by now)
# Wait until vLLM is up:
until curl -sf http://localhost:8000/v1/models; do sleep 10; done
echo "vLLM is ready"

curl http://localhost:8000/metrics | grep vllm | head -30
# screenshot: vllm_manual_query.png

# Confirm Prometheus target UP: http://localhost:9090/targets
# Confirm Grafana panels show data: http://localhost:3000

# ─── 1:00-1:30  Phase 1: flag justification (fill in REPORT.md)

# ─── 1:00-1:30  Download BIRD data + start agent server
python scripts/load_data.py  # downloads ~500MB, creates data/bird/*.sqlite, evals/eval_set.jsonl, load_test/perf_pool.jsonl
uvicorn agent.server:app --host 0.0.0.0 --port 8001 > ~/agent.log 2>&1 &
echo "Agent server starting on :8001. PID=$!"
until curl -sf http://localhost:8001/health; do sleep 3; done
echo "Agent ready"

# ─── 1:30-2:00  Baseline eval
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
python evals/run_eval.py --output results/eval_baseline.json
# screenshots: grafana_eval_run.png, langfuse_trace.png, langfuse_tags.png

# ─── 2:00-7:00  SLO tuning (see plan Phase B section 2:30-7:00)
python scripts/load_gen.py --rps 12 --duration 300 --base-url http://localhost:8000
# screenshots: grafana_before.png, grafana_after.png, grafana_serving.png
python evals/run_eval.py --output results/eval_after_tuning.json

# ─── 7:00-8:00  Screenshots, report, commit
# Work through screenshots/CHECKLIST.md
# Finalize REPORT.md
git add -A && git commit -m "H100 run results"
