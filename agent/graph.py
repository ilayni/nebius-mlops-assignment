from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("NEBIUS_API_KEY", "not-needed")


@dataclass
class AgentState:
    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


def _attach_schema(state: AgentState) -> dict:
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verify(text: str, execution: ExecutionResult) -> tuple[bool, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        ok_val = data.get("ok", True)
        if isinstance(ok_val, str):
            ok_val = ok_val.lower() == "true"
        return bool(ok_val), str(data.get("issue", ""))
    except (json.JSONDecodeError, AttributeError):
        return execution.ok, "Could not parse verifier response"


def generate_sql_node(state: AgentState) -> dict:
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("human", prompts.GENERATE_SQL_USER.format(
            schema=state.schema, question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("human", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            execution=state.execution.render() if state.execution else "No result",
        )),
    ])
    ok, issue = _parse_verify(response.content, state.execution)
    return {"verify_ok": ok, "verify_issue": issue}


def revise_node(state: AgentState) -> dict:
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("human", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            execution=state.execution.render() if state.execution else "No result",
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()


def run(question: str, db: str, question_id: str = "", db_label: str = "") -> dict:
    """Invoke the graph directly (no HTTP). db can be a db_id or an absolute file path.

    Returns a dict with: predicted_sql, result, final_status, iterations, latency_s,
    error, history.
    """
    state = AgentState(question=question, db_id=db)

    callbacks: list = []
    metadata: dict = {}
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler
            callbacks = [CallbackHandler()]
            if question_id:
                metadata["question_id"] = question_id
            if db_label:
                metadata["db"] = db_label
        except ImportError:
            pass

    t0 = time.monotonic()
    final = graph.invoke(state, config={"callbacks": callbacks, "metadata": metadata})
    latency = round(time.monotonic() - t0, 3)

    execution = final.get("execution")
    ok = bool(execution and execution.ok)

    return {
        "predicted_sql": final.get("sql", ""),
        "result": (execution.rows if ok else None),
        "final_status": "success" if ok else "error",
        "iterations": final.get("iteration", 0),
        "latency_s": latency,
        "error": (execution.error if execution and not execution.ok else None),
        "history": final.get("history", []),
    }
