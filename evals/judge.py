"""An LLM judge for the part the deterministic scorers cannot check: is the explanation right, and is it backed by evidence?

The judge gets the case's reference cause and the agent's report, never the Sentry or GitHub data, so it rates the
report against the reference, not against its own investigation. Scores are 1 to 5; the gate wants a mean of 3.5.
"""

import os

from mistralai.client import Mistral
from pydantic import BaseModel, Field

from report import InvestigationReport
from scoring import Case

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "mistral-medium-latest")

RUBRIC = """You grade an engineering investigation report against a reference explanation written by the engineer who planted the bug.

Score 1 to 5:
5 - The root cause matches the reference in substance (same defect, same violated assumption), names the responsible code,
    quotes or paraphrases the offending line, and cites Sentry and GitHub evidence.
4 - The root cause matches the reference but a detail is missing: no quoted line, thin evidence, or vague wording.
3 - The right file or function is named but the explanation is partly wrong, generic, or does not say which input violates which assumption.
2 - The report names the wrong code or a wrong mechanism, though it is in the right area.
1 - The report is wrong, empty, or claims evidence it does not cite.

Judge only what the report says. Do not reward length or confident tone. Do not penalise a report for not matching the reference's wording when the substance is the same."""


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5, description="1 to 5 per the rubric.")
    rationale: str = Field(description="One or two sentences: what the report got right and what it missed.")


def judge_prompt(case: Case, report: InvestigationReport) -> list[dict[str, str]]:
    """The messages the judge model gets."""
    user = (
        f"Question asked: {case.question}\n\n"
        f"Reference explanation: {case.reference_cause}\n"
        f"Reference location: {case.expected_file} :: {case.expected_symbol} ({case.expected_exception})\n\n"
        f"Report to grade (JSON):\n{report.model_dump_json(indent=2)}"
    )
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}]


async def judge(case: Case, report: InvestigationReport, *, client: Mistral | None = None) -> Verdict:
    """Rate `report` against the case's reference with the judge model."""
    client = client or Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = await client.chat.parse_async(response_format=Verdict, model=JUDGE_MODEL, messages=judge_prompt(case, report), temperature=0)
    choices = response.choices or []
    if not choices or choices[0].message.parsed is None:
        raise RuntimeError("The judge returned no parsable verdict")
    return choices[0].message.parsed
