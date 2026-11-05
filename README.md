## Phase 1 — Agent eval (20 tasks)

Plain-Python agent loop with 3 tools (calculator, sql_query, read_file), 6-step limit,
temperature 0, seed 42. Tasks in `evals/tasks.jsonl` (frozen), graded in code by `evals/run_eval.py`.

| Model | Variant | Passed | Avg steps | Bad calls | Wall time |
|---|---|---|---|---|---|
| qwen3:8b | baseline-v1 | 16/20 | 2.45 | 4 | 69 s |
| qwen3:4b | baseline-v1 | 17/20 | 2.5 | 13 | 1,613 s |

**Findings**
- All four qwen3:8b failures (t02, t04, t14, t17) need `notes.txt`. The agent can't discover
  filenames, so it guesses or skips the file. SQL- and calculator-only tasks: 100%.
- qwen3:4b writes its reasoning into the final answer despite `think=False`, so the
  number check is lenient on it. Treat 17/20 as an upper bound. 23× slower, 3× the bad calls.
- Grader v2: dates stripped before number extraction; "could not be found" accepted as a
  refusal. Both runs regraded from stored answers (no model re-run).