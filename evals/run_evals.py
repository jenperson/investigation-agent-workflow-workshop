"""Run the investigation agent on every case, score it, and apply the gate. This is the loop.

    uv run --env-file .env python evals/run_evals.py                 # every case, in-process agent
    uv run --env-file .env python evals/run_evals.py --cases 1,2     # a subset while you iterate
    uv run --env-file .env python evals/run_evals.py --served        # the agent `agents serve` or `agents deploy` runs
    uv run --env-file .env python evals/run_evals.py --no-judge      # deterministic scorers only, faster

Every run writes evals/results/<timestamp>.json and prints how it compares with the previous run. The exit code is 0
when the gate passes, 1 otherwise, so `make eval` can sit in CI.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mistralai.agents as agents

# The evals drive the agent in agent/ directly, so its modules must be importable from here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from investigate import investigate, served_agent
from judge import judge
from scoring import GATE, Case, CaseResult, Summary, load_cases, score, summarize

RESULTS_DIR = Path(__file__).with_name("results")
log = logging.getLogger("evals")


async def run_case(case: Case, *, agent: agents.Agent, use_judge: bool, semaphore: asyncio.Semaphore) -> CaseResult:
    """Investigate one case, score it, and judge it. Never raises: a failure is a result with `error` set."""
    async with semaphore:
        started = time.monotonic()
        try:
            outcome = await investigate(case.question, agent=agent)
        except Exception as error:
            log.warning("case %d failed: %s", case.id, error)
            return score(case, None, error=f"{type(error).__name__}: {error}")
        result = score(case, outcome.report, tool_calls=len(outcome.tool_calls), error=outcome.error, session_id=outcome.session_id)
        if use_judge and outcome.report is not None:
            try:
                verdict = await judge(case, outcome.report)
                result.judge, result.judge_rationale = verdict.score, verdict.rationale
            except Exception as error:
                log.warning("judge failed on case %d: %s", case.id, error)
        log.info("case %d done in %.0fs: file=%s symbol=%s cause=%s judge=%s", case.id, time.monotonic() - started, result.file, result.symbol, result.cause, result.judge)
        return result


async def run(cases: list[Case], *, agent: agents.Agent, use_judge: bool, concurrency: int) -> list[CaseResult]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*(run_case(case, agent=agent, use_judge=use_judge, semaphore=semaphore) for case in cases)))


def render_table(results: list[CaseResult], summary: Summary) -> str:
    """A readable table of the run, one row per case, then the aggregates and the gate."""
    mark = {True: "yes", False: "no"}
    rows = [f"{'case':>4}  {'ok':<3} {'file':<4} {'symbol':<6} {'exc':<4} {'cause':<5} {'judge':<5} {'calls':>5}  note"]
    for r in results:
        note = r.error or r.judge_rationale or ""
        judge = r.judge if r.judge is not None else "-"
        facts = f"{mark[r.file]:<4} {mark[r.symbol]:<6} {mark[r.exception]:<4} {mark[r.cause]:<5}"
        rows.append(f"{r.case_id:>4}  {mark[r.ok]:<3} {facts} {judge:<5} {r.tool_calls:>5}  {note[:70]}")
    rows.append("")
    judge_text = f"{summary.judge:.2f}" if summary.judge is not None else "-"
    facts = f"file {summary.file:.2f}  symbol {summary.symbol:.2f}  exception {summary.exception:.2f}  cause {summary.cause:.2f}"
    rows.append(f"completed {summary.completed}/{summary.cases}  {facts}  judge {judge_text}  mean tool calls {summary.mean_tool_calls:.1f}")
    gate = "  ".join(f"{k}>={v}" for k, v in GATE.items())
    rows.append(f"gate: {'PASS' if summary.passed else 'FAIL'} ({gate})" + ("" if summary.passed else "  failed: " + "; ".join(summary.failed_gates)))
    return "\n".join(rows)


def previous_run() -> dict[str, Any] | None:
    """The most recent saved run, if any."""
    runs = sorted(RESULTS_DIR.glob("*.json"))
    return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None


def render_delta(summary: Summary, previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "no previous run to compare with"
    before = previous["summary"]
    parts = []
    for name in ("file", "symbol", "exception", "cause", "judge"):
        now, then = getattr(summary, name), before.get(name)
        if now is None or then is None:
            continue
        parts.append(f"{name} {then:.2f} -> {now:.2f} ({now - then:+.2f})")
    return f"vs {previous['run_at']}: " + ", ".join(parts)


def save_run(results: list[CaseResult], summary: Summary, *, label: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    run_at = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = RESULTS_DIR / f"{run_at}.json"
    path.write_text(json.dumps({"run_at": run_at, "label": label, "summary": summary.to_dict(), "results": [r.to_dict() for r in results]}, indent=2), encoding="utf-8")
    return path


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", help="Comma-separated case IDs to run (default: all)")
    parser.add_argument("--served", action="store_true", help="Run against the served or deployed agent (INVESTIGATION_AGENT_NAME)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    parser.add_argument("--concurrency", type=int, default=3, help="Cases investigated at once (default 3)")
    parser.add_argument("--label", default="", help="A note saved with the run, such as 'instructions v3'")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    ids = {int(part) for part in args.cases.split(",")} if args.cases else None
    cases = load_cases(ids=ids)
    if not cases:
        raise SystemExit("No cases selected")
    if args.served:
        agent = served_agent()
    else:
        from main import agent

    previous = previous_run()
    results = await run(cases, agent=agent, use_judge=not args.no_judge, concurrency=args.concurrency)
    summary = summarize(results)
    print(render_table(results, summary))
    print(render_delta(summary, previous))
    print(f"saved {save_run(results, summary, label=args.label)}")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
