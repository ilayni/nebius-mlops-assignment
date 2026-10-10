Before you start: you don't need an H100 to do most of the development. Build and test the full pipeline locally first - the agent, evals, Langfuse tracing, vLLM with a small model. Point the agent at any OpenAI-compatible API while developing.

Once everything is working end-to-end locally, book an H100 slot and run the full pipeline against the 30B model. All screenshots and results in your submission must come from the H100 run.

https://github.com/GlebBerjoskin/mlops-assignment.git

❗Important: do all your development locally first. The full pipeline - agent, evals, Langfuse, even vLLM with a small model - can run on your laptop. Once everything works end-to-end, you book an H100 slot and run it against the 30B model. That's what you submit.


the full assignment instrunctions are in https://github.com/GlebBerjoskin/mlops-assignment

questions: what Nebius resources will we have available on the submissison tenant?
for example  [ ] Nebius object storage (for pre-loading model weights)
we can pre-download the Qwen3-30B weights to S3-compatible storage so the H100 doesn't spend its hours downloading.
on phase1 the repo says :
Configure the backend via VLLM_BASE_URL / VLLM_MODEL / OPENAI_API_KEY in .env (see the commented block there). 
why should we use openai api key ? should this be nebius api key?
answers:
Download took me ~2.5 min (on a vm) and is not the bottleneck (no need for object storage). The first-time warmup and kernel compilation, took me ~35 min total, is the time consuming part. but this is cached and saved after the first run. so subsequent skip this step and are much faster. Plan accordingly - start vLLM first thing when you spin up the VM, let it warm up, and don't restart it unnecessarily.
OPENAI_API_KEY is just a generic placeholder. If you're running vLLM locally, you don't need a real key. If you're using an external OpenAI-compatible service (like Token Factory), put the relevant (ToFa) API key there instead.