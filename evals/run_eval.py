"""Run the frozen agent eval (evals/tasks.jsonl) against one model and grade it in code.

Usage:
    uv run python evals/run_eval.py --model qwen3:8b                 # full run
    uv run python evals/run_eval.py --model qwen3:8b --limit 3       # smoke test
    uv run python evals/run_eval.py --model qwen3:8b --variant v2    # after a fix

Writes evals/results/<model>-<variant>-<YYYY-MM-DD>.json and prints passed/total.
Each task's full trace also goes to runs/*.jsonl via run_agent (look it up by task_id).
"""
import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ai_lab.agent import run_agent

EVAL_DIR = Path(__file__).resolve().parent
TASKS_PATH = EVAL_DIR / "tasks.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Same sampling settings for every model, so score differences are the model, not luck.
OPTIONS = {"temperature": 0, "seed": 42}

REL_TOL = 0.005   # number check: within 0.5% of expected...
ABS_TOL = 0.01    # ...or within 0.01, so expected=0 still works

REFUSE_PHRASES = (
    "not available", "don't have", "do not have", "don't know", "cannot", "can't",
    "unable", "not allowed", "only select", "read-only", "not found", "no such",
    "does not exist", "doesn't exist", "couldn't find", "could not find", "no email",
)

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def load_tasks(path: Path = TASKS_PATH) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task ids in {path}")
    return tasks


def extract_numbers(text: str) -> list[float]:
    """Every number in the text. Commas are removed first, so 348,000 and 3,48,000 both work."""
    return [float(n) for n in NUM.findall(text.replace(",", ""))]


def grade(task: dict, run: dict) -> tuple[bool, str]:
    """Return (passed, reason). The reason is what you read when a task fails."""
    if run["outcome"] != "answered":
        return False, run["outcome"]  # step_limit / empty_answer is always a fail

    answer = run["answer"]
    check = task["check"]

    if check == "number":
        expected = float(task["expected"])
        tol = max(REL_TOL * abs(expected), ABS_TOL)
        found = extract_numbers(answer)
        if any(abs(n - expected) <= tol for n in found):
            return True, "ok"
        return False, f"expected {expected:g}, found {[f'{n:g}' for n in found] or 'no numbers'}"

    if check == "contains":
        if task["expected"].lower() in answer.lower():
            return True, "ok"
        return False, f"missing '{task['expected']}'"

    if check == "refuse":
        if any(p in answer.lower() for p in REFUSE_PHRASES):
            return True, "ok"
        return False, "no refusal phrase (possible made-up answer)"

    raise ValueError(f"unknown check type '{check}' in task {task['id']}")


def git_commit() -> str | None:
    """Short hash of the code being evaluated, so every result maps back to a commit."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=EVAL_DIR).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval against one model.")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. qwen3:8b")
    parser.add_argument("--variant", default="baseline-v1", help="label for this agent version")
    parser.add_argument("--limit", type=int, help="only run the first N tasks (smoke test)")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Eval: {len(tasks)} tasks | model={args.model} | variant={args.variant} | options={OPTIONS}\n")
    t0 = time.perf_counter()
    results = []

    for task in tasks:
        try:
            run = run_agent(task["question"], model=args.model, verbose=False,
                            task_id=task["id"], options=OPTIONS)
            passed, reason = grade(task, run)
        except Exception as e:  # e.g. Ollama not running; record it instead of losing the whole run
            run = {"run_id": None, "outcome": "crash", "answer": None, "steps": 0,
                   "bad_calls": 0, "duration_s": 0.0, "tool_calls": []}
            passed, reason = False, f"crash: {type(e).__name__}: {e}"

        results.append({
            "id": task["id"],
            "category": task["category"],
            "passed": passed,
            "reason": reason,
            "outcome": run["outcome"],
            "steps": run["steps"],
            "tool_calls": len(run["tool_calls"]),
            "bad_calls": run["bad_calls"],
            "duration_s": run["duration_s"],
            "run_id": run["run_id"],
            "answer": run["answer"],
        })
        mark = "PASS" if passed else "FAIL"
        print(f"{task['id']}  {mark}  {run['steps']} steps  {run['bad_calls']} bad  "
              f"{run['duration_s']:>6.1f}s  {task['category']:<12} {'' if passed else reason}")

    passed_n = sum(r["passed"] for r in results)
    total = len(results)

    # per-category breakdown: tells you *what kind* of task the model struggles with
    by_cat: dict[str, list[int]] = {}
    for r in results:
        p, t = by_cat.setdefault(r["category"], [0, 0])
        by_cat[r["category"]] = [p + r["passed"], t + 1]

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": args.model,
        "variant": args.variant,
        "options": OPTIONS,
        "commit": git_commit(),
        "passed": passed_n,
        "total": total,
        "accuracy": round(passed_n / total, 3) if total else 0.0,
        "avg_steps": round(sum(r["steps"] for r in results) / total, 2) if total else 0.0,
        "total_bad_calls": sum(r["bad_calls"] for r in results),
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "by_category": {c: f"{p}/{t}" for c, (p, t) in sorted(by_cat.items())},
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.model.replace(':', '-')}-{args.variant}-{summary['date']}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBy category: {summary['by_category']}")
    print(f"Avg steps {summary['avg_steps']} | bad calls {summary['total_bad_calls']} | "
          f"wall time {summary['wall_time_s']}s")
    print(f"Saved {out.relative_to(EVAL_DIR.parent)}")
    print(f"\n{args.model} [{args.variant}]: {passed_n}/{total}")


if __name__ == "__main__":
    main()