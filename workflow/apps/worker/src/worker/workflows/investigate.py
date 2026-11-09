"""The `investigate` workflow: a durable wrapper around one agent investigation.

Start it from Studio (Workflows -> investigate -> Start Workflow) or the API with `{"question": "..."}`. The execution
records its input, the activity and the result, and survives a worker restart. A Slack trigger or a schedule can start
it the same way (see workshop/05-stretch.md).

The agent runs elsewhere, under `agents serve` on your machine or `agents deploy` on Mistral; the activity only opens
a session on it by name and runs one turn. The Agents SDK is imported inside the activity, where network I/O belongs:
workflow code runs in a determinism sandbox that rejects it at module level.
"""

import json
import os
from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from pydantic import BaseModel, Field

DEFAULT_AGENT_NAME = "investigation-agent"
TURN_TIMEOUT_SECONDS = 600


class InvestigateInput(BaseModel):
    """What starts an investigation."""

    question: str = Field(description='The question to investigate, such as "What\'s causing the KeyError in checkout?"')


class InvestigateResult(BaseModel):
    """What an investigation produced."""

    report: dict[str, Any] = Field(description="The agent's InvestigationReport as JSON.")
    text: str = Field(description="The report rendered as plain text, ready to post in a chat.")
    session_id: str = Field(description="The agent session the investigation ran in, for the Studio session view.")
    tool_calls: list[str] = Field(default_factory=list, description="The tools the agent called, in order.")


def render_report(report: dict[str, Any]) -> str:
    """The report as plain text with no Markdown tables, so it reads well in Slack, a terminal, or Studio."""
    lines = [report.get("summary", "").strip(), ""]
    if report.get("exception"):
        lines.append(f"Exception: {report['exception']}")
    where = " :: ".join(part for part in (report.get("file"), report.get("symbol")) if part)
    if where:
        lines.append(f"Where: {where}")
    lines += [f"Confidence: {report.get('confidence', 'unknown')}", "", "Root cause", report.get("root_cause", "").strip()]
    if report.get("evidence"):
        lines += ["", "Evidence"]
        lines += [f"- [{e.get('source')}] {e.get('reference')}: {e.get('proves')}" for e in report["evidence"]]
    if report.get("next_check"):
        lines += ["", f"Next check: {report['next_check']}"]
    return "\n".join(lines).strip()


def agent_name() -> str:
    """The name the agent is served or deployed under: INVESTIGATION_AGENT_NAME, else the default."""
    return os.environ.get("INVESTIGATION_AGENT_NAME", "").strip() or DEFAULT_AGENT_NAME


def read_report(state: object) -> dict[str, Any]:
    """The InvestigationReport JSON of a completed turn, or a RuntimeError saying why there is none."""
    import mistralai.agents as agents

    match state:
        case agents.Completed(text=text, stop_reason=None):
            try:
                return json.loads(text)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"The agent's answer is not JSON: {error}") from error
        case agents.Completed(stop_reason=reason):
            raise RuntimeError(f"The agent turn ended without a final answer ({reason})")
        case agents.Failed(error=error):
            raise RuntimeError(f"The agent turn failed ({error.code}): {error.message}")
        case agents.Blocked():
            raise RuntimeError("The agent turn waits on an approval that a workflow cannot answer")
        case _:
            raise RuntimeError(f"The agent turn did not complete: {type(state).__name__}")


# Not retried: a second attempt would start a second investigation on a new session. The turn's idempotency key is the
# workflow execution ID, so a replayed activity joins the turn that is already running instead.
@workflows.activity(start_to_close_timeout=timedelta(seconds=TURN_TIMEOUT_SECONDS + 60), retry_policy_max_attempts=1)
async def run_investigation(question: str, idempotency_key: str) -> dict[str, Any]:
    """Open a session on the served or deployed agent, run `question` as one turn, and return the report with its context."""
    import mistralai.agents as agents

    agent = agents.Agent.load_from(agent_name=agent_name())
    session = await agent.start_session()
    try:
        state = await session.run(question, idempotency_key=idempotency_key, timeout=TURN_TIMEOUT_SECONDS)
        report = read_report(state)
        return {"report": report, "session_id": session.id, "tool_calls": [call.name for call in getattr(state, "tool_calls", ())]}
    finally:
        await session.cancel()


@workflows.workflow.define(
    name="investigate",
    workflow_display_name="Investigate an error",
    workflow_description="Ask the investigation agent a question about a production error and return its InvestigationReport.",
)
class Investigate:
    @workflows.workflow.entrypoint
    async def run(self, input: InvestigateInput) -> InvestigateResult:
        # The execution ID keys the agent turn, so a replayed activity never starts a second investigation.
        outcome = await run_investigation(input.question, workflows.get_execution_id())
        return InvestigateResult(
            report=outcome["report"],
            text=render_report(outcome["report"]),
            session_id=outcome["session_id"],
            tool_calls=outcome["tool_calls"],
        )
