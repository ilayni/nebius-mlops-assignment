# H100 Session Plan

**Total window: 8 hours.** vLLM first-time warmup is 35 min — start it in minute 1, not minute 10.

---

## Step 0 — Create and Configure the VM (do this BEFORE 15:00)

### 0.1 Create the VM in Nebius Console

1. Open Nebius Console → **Compute Cloud** → **Virtual Machines** → **Create VM**
2. **Platform / preset:** GPU — select the H100 80 GB preset (1× NVIDIA H100 SXM5 80 GB)
3. **Operating system:** `Ubuntu 22.04 LTS (CUDA 12.x pre-installed)` — pick the image that already has CUDA drivers; this saves ~20 min on driver setup
4. **Boot disk:**
   - **Type:** SSD (NVMe) — model weights read at startup; spinning disk or network disk will slow the first load significantly
   - **Size: 300 GB** — breakdown: model weights ~70 GB, BIRD data ~1 GB extracted, Docker images ~8 GB, OS + packages ~10 GB, headroom for KV cache spill + results
5. **Network:** leave defaults (public IP needed for GitHub clone and HuggingFace download)

### 0.2 Add Your SSH Public Key

In the same VM creation form, under **SSH key** / **Access**:

Paste the contents of `~/.ssh/nebius_ed25519.pub`:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIML30In7Sin1cae8VyfC0mKl7EcyoMxQEk92TXfasKmZ nebius cloud ssh key
```

If the VM already exists and you missed this step, use:
**Console → VM → Edit → SSH keys → Add key**

### 0.3 Note the VM IP

Once the VM is `Running`, copy the **public IP** from the Console. You'll need it in every command below. Export it locally:

```bash
export H100_IP=<paste IP here>
```

---

## Step 1 — Open SSH Tunnel + Clone (minute 0–2)

Run this **on your Mac** in a dedicated terminal. Keep it open the entire session — closing it kills access to Grafana/Langfuse/vLLM.

```bash
ssh -i ~/.ssh/nebius_ed25519 \
    -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    ubuntu@$H100_IP
```

> Username is `ubuntu` on Ubuntu images. If it differs, check the Console under Access.

Once inside the VM:

```bash
git clone git@github.com:ilayni/nebius-mlops-assignment.git app
cd app
```

If GitHub SSH isn't set up on the VM yet, use HTTPS:
```bash
git clone https://github.com/ilayni/nebius-mlops-assignment.git app
cd app
```

---

## Step 2 — Start vLLM IMMEDIATELY (minute 2–3)

**Do this before anything else.** The 35-min warmup clock starts now.

```bash
mkdir -p ~/models/qwen3-30b

# Download weights first (~2.5 min, ~70 GB)
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct --local-dir ~/models/qwen3-30b

# Start vLLM in background
nohup bash scripts/start_vllm.sh > ~/vllm.log 2>&1 &
echo "vLLM PID=$! — warmup ~35 min. Monitor: tail -f ~/vllm.log"
```

---

## Step 3 — Install Dependencies + Start Observability Stack (minute 5–40)

Do all of this while vLLM warms up in the background.

### 3.1 Install system tools

```bash
sudo apt-get update -qq
sudo apt-get install -y docker.io docker-compose-v2 python3-pip python3-dev
sudo usermod -aG docker $USER
newgrp docker          # pick up docker group without re-login
```

### 3.2 Install Python deps

```bash
pip install uv
uv pip install --system -r requirements.txt
# or: pip install -r requirements.txt
```

### 3.3 Configure `.env`

```bash
cp .env.example .env
nano .env
```

Set these values (everything else can stay as-is):

```
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-11111111-2222-3333-4444-555555555555
LANGFUSE_SECRET_KEY=sk-lf-66666666-7777-8888-9999-aaaaaaaaaaaa
```

### 3.4 Start Prometheus + Grafana + Langfuse

```bash
docker compose up -d
```

Wait ~60 s for postgres healthcheck, then verify:

```bash
curl -s http://localhost:3001/api/public/health   # → {"status":"ok"}
```

URLs (via SSH tunnel on your Mac browser):
- Grafana: http://localhost:3000  (admin / admin)
- Prometheus: http://localhost:9090
- Langfuse: http://localhost:3001  (admin@local.dev / admin123)

### 3.5 Download BIRD data

```bash
python scripts/load_data.py
# Downloads ~500 MB, extracts SQLite DBs, writes evals/eval_set.jsonl and load_test/perf_pool.jsonl
```

### 3.6 Start the agent server

```bash
uvicorn agent.server:app --host 0.0.0.0 --port 8001 > ~/agent.log 2>&1 &
until curl -sf http://localhost:8001/health; do sleep 3; done
echo "Agent ready"
```

---

## Step 4 — Validate Full Stack (minute 40–60)

By now vLLM should be finishing warmup. Wait for it:

```bash
until curl -sf http://localhost:8000/v1/models; do
  echo "waiting for vLLM..."; sleep 10
done
echo "vLLM ready"
```

Run all health checks:

```bash
# vLLM
curl http://localhost:8000/v1/models
curl -s http://localhost:8000/metrics | grep vllm_request | head -10

# Agent
curl -s -X POST http://localhost:8001/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many rows in any table?","db":"test"}' | python3 -m json.tool

# Prometheus
curl -s http://localhost:9090/-/ready   # → Prometheus is Ready.

# Langfuse
curl -s http://localhost:3001/api/public/health
```

**Screenshot now:** `screenshots/vllm_manual_query.png`
- Browser: open http://localhost:8000/docs → try a completion request, screenshot the response

Confirm in Prometheus (http://localhost:9090/targets) that the vLLM target is **UP**.
Confirm Grafana panels (http://localhost:3000) show data.

---

## Step 5 — Phase 1: vLLM Flag Justification (minute 60–90)

Fill in `REPORT.md` — explain each flag passed to vLLM:

| Flag | Value | Reason |
|---|---|---|
| `--dtype` | `bfloat16` | Qwen3-30B ships in bfloat16; avoids re-quantisation on H100 |
| `--tensor-parallel-size` | `1` | Single GPU — TP>1 adds NCCL overhead with no benefit |
| `--gpu-memory-utilization` | `0.90` | Leaves ~8 GB for KV-cache bursts; 0.95+ risks OOM |
| `--max-model-len` | `8192` | SQL+schema fits in 8k tokens; shorter context = larger KV cache |
| `--enable-chunked-prefill` | — | Prevents long prefills from starving decode; keeps P95 stable |

---

## Step 6 — Baseline Eval (minute 90–120)

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
python evals/run_eval.py --output results/eval_baseline.json
```

**Screenshots:**
- `screenshots/grafana_eval_run.png` — Grafana during the eval run (request rate panel)
- `screenshots/langfuse_trace.png` — one full trace in Langfuse showing generate→execute→verify nodes
- `screenshots/langfuse_tags.png` — Langfuse traces list with tags/metadata visible

---

## Step 7 — SLO Tuning (minute 120–420)

**SLO target:** P95 latency < 5 s at ≥ 10 RPS sustained over 5 min.

### 7.1 Establish before-state screenshot

Run a short load test, screenshot Grafana:

```bash
python scripts/load_gen.py --rps 10 --duration 60 --url http://localhost:8001/answer
```

**Screenshot:** `screenshots/grafana_before.png` — Grafana showing P95 before tuning

### 7.2 Tune vLLM parameters

Try these levers (DO NOT restart vLLM unless a setting requires it — each restart costs 5–35 min):

Settings you can tune without restarting:
- Adjust concurrency in the agent server (`--workers` in uvicorn)
- Tune `--max-num-seqs` at next restart if needed

Settings that require a vLLM restart (avoid if possible, each costs 5–35 min):
- `--max-num-seqs` (default 256) — increase to handle more parallel requests
- `--max-model-len` — reduce to 4096 if you don't need long context (frees KV cache)
- `--enable-prefix-caching` — add this flag to cache repeated schema prefixes across requests

To restart vLLM with new flags:
```bash
pkill -f vllm
sleep 5
nohup bash scripts/start_vllm.sh > ~/vllm.log 2>&1 &
until curl -sf http://localhost:8000/v1/models; do sleep 10; done
```

### 7.3 SLO validation load test (5 min at 12 RPS)

```bash
python scripts/load_gen.py --rps 12 --duration 300 --url http://localhost:8001/answer
```

**Screenshot:** `screenshots/grafana_after.png` — Grafana after tuning, P95 < 5s visible
**Screenshot:** `screenshots/grafana_serving.png` — Grafana serving dashboard overview

### 7.4 Post-tuning eval

```bash
python evals/run_eval.py --output results/eval_after_tuning.json
```

---

## Step 8 — Wrap Up (minute 420–480)

### 8.1 Complete all screenshots

Work through `screenshots/CHECKLIST.md`. Required:
- [ ] `vllm_manual_query.png`
- [ ] `grafana_serving.png`
- [ ] `langfuse_trace.png`
- [ ] `langfuse_tags.png`
- [ ] `grafana_eval_run.png`
- [ ] `grafana_before.png`
- [ ] `grafana_after.png`

Copy screenshots from your Mac browser (they're already local via the SSH tunnel — save directly to `screenshots/`).

### 8.2 Finalize REPORT.md

Fill in actual numbers from the H100 run (eval accuracy, P95 latency, RPS achieved).

### 8.3 Commit and push

```bash
git add results/ screenshots/ REPORT.md
git commit -m "H100 run results — eval + SLO screenshots"
git push
```

---

## Quick Reference

| Service | URL (via tunnel) | Credentials |
|---|---|---|
| vLLM API | http://localhost:8000 | no auth |
| Agent | http://localhost:8001 | no auth |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | no auth |
| Langfuse | http://localhost:3001 | admin@local.dev / admin123 |

| Log | Command |
|---|---|
| vLLM | `tail -f ~/vllm.log` |
| Agent | `tail -f ~/agent.log` |
| Docker stack | `docker compose logs -f` |

| Time | Activity |
|---|---|
| 0:00 | SSH in, clone repo |
| 0:02 | **Start vLLM (35-min warmup begins)** |
| 0:05 | Install deps, start Docker stack, download BIRD data |
| 0:40 | vLLM ready — validate all endpoints |
| 1:00 | Fill in REPORT.md flag justification |
| 1:30 | Baseline eval + screenshots |
| 2:00 | SLO tuning begins |
| 7:00 | Screenshots, finalize report, commit & push |
