"""The investigation agent: its model, its requirements (instructions.md), and what it must answer with.

The `agents` CLI loads `agent` from here (`agents chat`, `agents serve`, `agents deploy`). The tests build the same agent
through `build_agent(connectors=False)` and drive it with a scripted model, so they run offline.
"""

import os
from pathlib import Path

import mistralai.agents as agents

from report import InvestigationReport

ROOT = Path(__file__).resolve().parent
MODEL = "mistral-large-4"
# `agents serve --name` and `agents deploy --name` override it; the workflow opens sessions on this name.
DEFAULT_AGENT_NAME = "investigation-agent"


def build_agent(*, connectors: bool = True, name: str | None = None) -> agents.Agent:
    """Build the agent from its configuration. `connectors` has no effect yet: module 2 adds the connectors."""
    return agents.Agent(
        name=name or os.environ.get("INVESTIGATION_AGENT_NAME", "").strip() or DEFAULT_AGENT_NAME,
        model=MODEL,
        description="Investigates production errors and answers with an InvestigationReport.",
        instructions=(ROOT / "instructions.md").read_text(encoding="utf-8"),
        # Every turn ends with a validated InvestigationReport: the evals and the workflow read it as data.
        output_type=InvestigationReport,
        max_iterations=32,
        temperature=0.1,
    )


agent = build_agent()
