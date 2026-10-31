"""Plain-Python agent loop: model -> tool calls -> results -> model, until an answer or the step limit.

Run:
    uv run python -m ai_lab.agent             # demo question (needs 3 tool calls)
    uv run python -m ai_lab.agent selftest    # bad-call handling checks, no model needed
"""
import json
import sys

import ollama

from ai_lab.tools import TOOLS, TOOL_FUNCTIONS

MODEL = "qwen3:8b"
MAX_STEPS = 6  # one step = one model call

SYSTEM_PROMPT = (
    "You are an agent that answers questions using tools. "
    "Never guess numbers: get data from sql_query or read_file, and do arithmetic with calculator. "
    "You may call tools over several turns. If a tool returns an ERROR, read it, fix the call, and retry. "
    "When you have everything you need, reply with the final answer and make no tool call."
)

# name -> JSON schema of its parameters, so we can validate args before calling Python
TOOL_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def execute_tool_call(name: str, args) -> tuple[str, bool]:
    """Run one tool call. Returns (result_text, ok). Never raises.

    Every failure becomes an ERROR string that goes back to the model as the tool result,
    so the model can see what went wrong and correct itself on the next step.
    """
    if name not in TOOL_FUNCTIONS:
        return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(TOOL_FUNCTIONS)}.", False

    if args is None:
        args = {}
    if isinstance(args, str):  # some models send arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"ERROR: arguments for {name} are not valid JSON: {args!r}", False
    if not isinstance(args, dict):
        return f"ERROR: arguments for {name} must be a JSON object, got {type(args).__name__}.", False

    schema = TOOL_SCHEMAS[name]
    missing = [p for p in schema.get("required", []) if p not in args]
    unexpected = [k for k in args if k not in schema.get("properties", {})]
    if missing or unexpected:
        expected = ", ".join(schema.get("properties", {}))
        return (f"ERROR: bad arguments for {name}. Missing: {missing or 'none'}. "
                f"Unexpected: {unexpected or 'none'}. Expected parameters: {expected}."), False

    try:
        return str(TOOL_FUNCTIONS[name](**args)), True
    except Exception as e:  # tool-level failure: bad SQL, file not found, unsupported expression...
        return f"ERROR: {name} failed: {type(e).__name__}: {e}", False


def run_agent(question: str, model: str = MODEL, max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    """Run the loop for one question. Returns a plain dict (ready to log as one JSONL line later)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_log = []

    def result(outcome: str, answer, steps: int) -> dict:
        return {"question": question, "model": model, "outcome": outcome,
                "answer": answer, "steps": steps, "tool_calls": tool_log}

    for step in range(1, max_steps + 1):
        try:
            response = ollama.chat(model=model, messages=messages, tools=TOOLS, think=False)
        except ollama.ResponseError as e:
            # Ollama couldn't parse the model's tool-call output. Tell the model and spend the step.
            if verbose:
                print(f"[step {step}] model output rejected by Ollama: {e}")
            messages.append({"role": "user", "content":
                             f"Your last response could not be parsed ({e}). "
                             "Make a valid tool call or give the final answer."})
            continue

        msg = response.message
        messages.append(msg)  # keep the model's turn (including its tool requests) in history

        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            if verbose:
                print(f"[step {step}] final answer")
            return result("answered" if answer else "empty_answer", answer, step)

        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            output, ok = execute_tool_call(name, args)
            if verbose:
                tag = "ok" if ok else "BAD"
                print(f"[step {step}] {name}({args}) -> {tag}: {output[:200]}")
            tool_log.append({"step": step, "tool": name, "args": args, "ok": ok, "result": output[:500]})
            messages.append({"role": "tool", "content": output, "tool_name": name})

    if verbose:
        print(f"[stopped] hit the {max_steps}-step limit without a final answer")
    return result("step_limit", None, max_steps)


def selftest() -> None:
    """Deterministic checks for bad-call handling. No model involved."""
    cases = [
        ("unknown tool",      "web_search",  {"q": "x"},                     False),
        ("missing arg",       "calculator",  {},                             False),
        ("unexpected arg",    "calculator",  {"expr": "1+1"},                False),
        ("args not JSON",     "calculator",  "{expression: 1+1",             False),
        ("args wrong type",   "calculator",  ["1+1"],                        False),
        ("tool raises",       "sql_query",   {"query": "DELETE FROM employees"}, False),
        ("bad expression",    "calculator",  {"expression": "__import__('os')"}, False),
        ("path escape",       "read_file",   {"path": "../company.db"},      False),
        ("args as JSON str",  "calculator",  '{"expression": "2*21"}',       True),
        ("good call",         "calculator",  {"expression": "(10+5)*2"},     True),
    ]
    failed = 0
    for label, name, args, expect_ok in cases:
        output, ok = execute_tool_call(name, args)
        status = "PASS" if ok == expect_ok else "FAIL"
        failed += status == "FAIL"
        print(f"{status}  {label:18} -> {output[:90]}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        return

    question = (sys.argv[1] if len(sys.argv) > 1 else
                "Read notes.txt to find how many engineers we plan to hire in Q4. "
                "If each new engineer is paid the current average Engineering salary, "
                "what is the total yearly cost of the new engineers?")
    print(f"Q: {question}\n")
    run = run_agent(question)
    print(f"\nOutcome: {run['outcome']} in {run['steps']} step(s), {len(run['tool_calls'])} tool call(s)")
    print(f"Answer: {run['answer']}")


if __name__ == "__main__":
    main()