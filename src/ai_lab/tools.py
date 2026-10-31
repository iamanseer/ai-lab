import ast
import json
import operator
import sqlite3
from pathlib import Path

# tools.py is at src/ai_lab/tools.py, so parents[2] is the repo root
SANDBOX = Path(__file__).resolve().parents[2] / "sandbox"
DB_PATH = SANDBOX / "company.db"
FILES_DIR = SANDBOX / "files"

# --- calculator: safe arithmetic, never use eval() on model output ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")

def calculator(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))

# --- sql_query: read-only SELECT on the sandbox DB ---
def sql_query(query: str) -> str:
    if not query.strip().lower().startswith("select"):
        raise ValueError("only SELECT queries are allowed")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.execute(query)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(50)
    finally:
        conn.close()
    return json.dumps({"columns": columns, "rows": rows})

# --- read_file: only inside sandbox/files ---
def read_file(path: str) -> str:
    target = (FILES_DIR / path).resolve()
    if not target.is_relative_to(FILES_DIR.resolve()):
        raise ValueError("path is outside the sandbox")
    return target.read_text()[:4000]

# What the model sees
TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Supports + - * / ** % and parentheses.",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '(12.5 * 4) / 3'"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "sql_query",
        "description": ("Run a read-only SQLite SELECT on the company database. "
                        "Table: employees(id INTEGER, name TEXT, department TEXT, salary INTEGER, joined DATE)."),
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT statement"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the sandbox files folder. Use a relative path like 'notes.txt'.",
        "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]}}},
]

# Your dict-dispatch pattern from Phase 0
TOOL_FUNCTIONS = {"calculator": calculator, "sql_query": sql_query, "read_file": read_file}