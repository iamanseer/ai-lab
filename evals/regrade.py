"""Re-grade saved eval results with the current grade() — no model calls.

Use this only when the GRADER changes (a bug in how answers are checked).
The model's answers stay exactly as recorded; only pass/fail is recomputed.
Never use it after changing tasks.jsonl expected values: those are frozen.

Usage:
    uv run python evals/regrade.py evals/results/qwen3-8b-baseline-v1-2026-11-05.json [more files...]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from run_eval import grade, load_tasks  # same folder, so the same grader the runner uses


def regrade(path: Path, tasks: dict[str, dict]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    before = data["passed"]

    for r in data["results"]:
        if r["outcome"] == "crash":
            continue  # nothing to grade
        passed, reason = grade(tasks[r["id"]], r)
        if passed != r["passed"]:
            print(f"  {r['id']}: {'FAIL -> PASS' if passed else 'PASS -> FAIL'}  ({reason})")
        r["passed"], r["reason"] = passed, reason

    by_cat: dict[str, list[int]] = {}
    for r in data["results"]:
        p, t = by_cat.get(r["category"], [0, 0])
        by_cat[r["category"]] = [p + r["passed"], t + 1]

    data["passed"] = sum(r["passed"] for r in data["results"])
    data["accuracy"] = round(data["passed"] / data["total"], 3)
    data["by_category"] = {c: f"{p}/{t}" for c, (p, t) in sorted(by_cat.items())}
    data["regraded_at"] = datetime.now().isoformat(timespec="seconds")

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{path.name}: {before}/{data['total']} -> {data['passed']}/{data['total']}  {data['by_category']}\n")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tasks = {t["id"]: t for t in load_tasks()}
    for p in sys.argv[1:]:
        regrade(Path(p), tasks)


if __name__ == "__main__":
    main()