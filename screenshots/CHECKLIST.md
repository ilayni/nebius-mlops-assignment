# Screenshot Checklist

Check off each item **before the final commit** on H100.
(Note: the assignment plan originally mentioned 11 screenshots; the submission checklist lists 7 named files.
Reconcile against the actual assignment repo if there is a discrepancy.)

- [ ] `vllm_manual_query.png` — vLLM serving + manual curl query showing SQL response (step B 0:40-1:00)
- [ ] `grafana_eval_run.png` — Grafana dashboard during baseline eval run (step B 1:30-2:00)
- [ ] `langfuse_trace.png` — Langfuse trace showing generate_sql → verify → revise waterfall
- [ ] `langfuse_tags.png` — Langfuse trace list with metadata tags visible
- [ ] `grafana_before.png` — Grafana metrics **before** SLO tuning
- [ ] `grafana_after.png` — Grafana metrics **after** SLO tuning
- [ ] `grafana_serving.png` — Full Grafana dashboard with all panels under load
