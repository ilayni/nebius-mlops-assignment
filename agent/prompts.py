GENERATE_SQL_SYSTEM = (
    "You are a SQL expert. Write a single SQLite SQL query that answers the question. "
    "Return ONLY the SQL — no explanation, no markdown."
)

GENERATE_SQL_USER = "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

VERIFY_SYSTEM = (
    "You are a SQL result checker. Decide if the execution result plausibly answers the question. "
    'Respond ONLY with JSON: {"ok": true, "issue": ""} or {"ok": false, "issue": "reason"}.'
)

VERIFY_USER = "Question: {question}\n\nSQL: {sql}\n\nResult: {execution}\n\nJSON:"

REVISE_SYSTEM = (
    "You are a SQL expert. Fix the SQL query that failed verification. "
    "Return ONLY the corrected SQL — no explanation, no markdown."
)

REVISE_USER = (
    "Schema:\n{schema}\n\nQuestion: {question}\n\n"
    "Failed SQL:\n{sql}\n\nExecution:\n{execution}\n\nIssue: {issue}\n\nCorrected SQL:"
)
