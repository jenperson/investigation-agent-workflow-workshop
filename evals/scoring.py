"""The cases, the scorers, and the gate: pure functions, so they are unit-tested and the same in every run.

A case says what a correct investigation must find. The deterministic scorers check the facts (file, symbol, exception,
root-cause keywords); the LLM judge in `judge.py` rates the explanation. `summarize()` turns a run into the numbers the
gate is applied to.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from report import InvestigationReport

CASES_PATH = Path(__file__).with_name("cases.jsonl")

# The gate: what "good enough" means for this agent. Raise it as the instructions improve.
GATE: dict[str, float] = {"file": 0.8, "symbol": 0.8, "exception": 0.8, "cause": 0.8, "judge": 3.5}
# How many of a case's keywords the root cause must mention.
MIN_KEYWORD_HITS = 2


@dataclass(frozen=True)
class Case:
    id: int
    question: str
    expected_exception: str
    expected_file: str
    expected_symbol: str
    cause_keywords: tuple[str, ...]
    reference_cause: str


def load_cases(path: Path = CASES_PATH, ids: set[int] | None = None) -> list[Case]:
    """Read the cases, optionally only those in `ids`."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["cause_keywords"] = tuple(raw["cause_keywords"])
        case = Case(**raw)
        if ids is None or case.id in ids:
            cases.append(case)
    return cases


def normalize_path(path: str) -> list[str]:
    """Path components, ignoring leading './', '/' and a repository prefix such as 'owner/repo/blob/main/'."""
    parts = [part for part in path.replace("\\", "/").split("/") if part and part != "."]
    if "blob" in parts:
        parts = parts[parts.index("blob") + 2 :]
    return parts


def file_matches(expected: str, actual: str | None) -> bool:
    """True when `actual` names the expected file: the same trailing path components, at least the file name and its folder."""
    if not actual:
        return False
    want, got = normalize_path(expected), normalize_path(actual)
    if not got or got[-1] != want[-1]:
        return False
    overlap = min(len(want), len(got))
    return want[-overlap:] == got[-overlap:] and overlap >= min(2, len(want))


def symbol_matches(expected: str, actual: str | None) -> bool:
    """True when `actual` names the expected function, with or without a module or class prefix and call parentheses."""
    if not actual:
        return False
    return actual.strip().removesuffix("()").split(".")[-1] == expected


def exception_matches(expected: str, actual: str | None) -> bool:
    """True when the reported exception names the expected type."""
    return bool(actual) and expected.lower() in actual.lower()


def cause_matches(keywords: tuple[str, ...], text: str, minimum: int = MIN_KEYWORD_HITS) -> bool:
    """True when the root cause mentions at least `minimum` of the case's keywords (case-insensitive substrings)."""
    hay = text.lower()
    return sum(keyword.lower() in hay for keyword in keywords) >= min(minimum, len(keywords))


@dataclass
class CaseResult:
    """How one case went: the report (if any), the deterministic scores, and the judge's rating."""

    case_id: int
    question: str
    ok: bool
    file: bool = False
    symbol: bool = False
    exception: bool = False
    cause: bool = False
    judge: int | None = None
    judge_rationale: str = ""
    tool_calls: int = 0
    error: str | None = None
    report: dict[str, Any] | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score(case: Case, report: InvestigationReport | None, *, tool_calls: int = 0, error: str | None = None, session_id: str | None = None) -> CaseResult:
    """Apply the deterministic scorers to one report. A missing report scores zero on everything."""
    result = CaseResult(case_id=case.id, question=case.question, ok=report is not None, tool_calls=tool_calls, error=error, session_id=session_id)
    if report is None:
        return result
    result.report = report.model_dump()
    result.file = file_matches(case.expected_file, report.file)
    result.symbol = symbol_matches(case.expected_symbol, report.symbol)
    result.exception = exception_matches(case.expected_exception, report.exception)
    result.cause = cause_matches(case.cause_keywords, f"{report.summary}\n{report.root_cause}")
    return result


@dataclass
class Summary:
    """The run's numbers: one accuracy per deterministic scorer, the mean judge score, and whether the gate passed."""

    cases: int
    completed: int
    file: float
    symbol: float
    exception: float
    cause: float
    judge: float | None
    mean_tool_calls: float
    failed_gates: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed_gates

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def summarize(results: list[CaseResult], gate: dict[str, float] = GATE) -> Summary:
    """Aggregate the results and apply the gate. The judge gate is skipped when no case was judged."""
    if not results:
        raise ValueError("No results to summarize")
    judged = [r.judge for r in results if r.judge is not None]
    summary = Summary(
        cases=len(results),
        completed=sum(r.ok for r in results),
        file=mean(r.file for r in results),
        symbol=mean(r.symbol for r in results),
        exception=mean(r.exception for r in results),
        cause=mean(r.cause for r in results),
        judge=mean(judged) if judged else None,
        mean_tool_calls=mean(r.tool_calls for r in results),
    )
    for name, threshold in gate.items():
        value = getattr(summary, name)
        if value is not None and value < threshold:
            summary.failed_gates.append(f"{name} {value:.2f} < {threshold:.2f}")
    return summary
