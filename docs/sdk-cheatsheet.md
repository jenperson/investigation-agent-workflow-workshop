# Mistral Agents SDK 1.1 cheat sheet

The exact API surface this repository uses, verified against `mistralai-agents==1.1.3`. Point your coding agent at
this file before it writes agent code: the SDK is young, 1.1 renamed several keywords from the release candidates, and
the training data of most models predates it. Every name here was checked against the installed package, so build from
this file as written; introspect the package only if a test fails with a `TypeError` or `AttributeError` on an SDK call.

## Import

```python
import mistralai.agents as agents
```

## Define an agent

```python
agent = agents.Agent(
    name="investigation-agent",  # the name `agents serve` / `agents deploy` register; no dots
    model="mistral-large-4",
    description="One sentence for the catalog.",
    instructions=open("instructions.md").read(),  # the system prompt
    output_type=InvestigationReport,  # a Pydantic model; every turn must end with it
    max_iterations=32,  # cap on loop steps per turn
    temperature=0.1,
    environment=agents.BashkitSandbox(access="none"),  # the sandbox (`sandbox=` in the 1.1 release candidates)
    skills=[agents.LocalSkillBundle.from_directory(path) for path in sorted(SKILLS_DIR.iterdir())],
    connectors=[agents.connector("sentry_d183", run_as="auto", auto_auth=False)],
    permission_policy=agents.PermissionPolicy(default="deny", rules=(...,)),
    hooks=[AuditTrail()],
    builtins=agents.Builtins(subagents=False),
)
```

Facts that bite:

- `instructions=` replaces the runtime's default identity prompt. Read the file yourself; `AGENTS.md` has no automatic meaning.
- **Skills need a sandbox that can hold files.** The default `NoSandbox()` refuses skills. `BashkitSandbox(access="none")` holds the skill files and gives the model **no** shell or file tools (needs the `bashkit` extra: `mistralai-agents[bashkit]`).
- `agents.connector(name)` names a connector of the Studio workspace. The SDK puts its tools under the namespace `connector_<name>` where every non-alphanumeric character becomes `_` (e.g. `connector_sentry_d183.search_issues`). **Connectors need programmatic tool calling**, so leave `Builtins(programmatic_tool_calling=True)` (the default).
- Connectors are resolved when a session starts, under the identity of the session user (`MISTRAL_API_KEY` in-process). Building the `Agent` does not touch the network, so tests can build it offline and run it with no connectors.
- `agent.connectors`, `agent.skills`, `agent.hooks` are **not** public attributes. Public ones: `name`, `model`, `instructions`, `output_type`, `permission_policy`, `max_iterations`, `temperature`, `builtins`, `description`.

## Permission policy (deny by default)

```python
agents.PermissionPolicy(
    default="deny",
    rules=(
        agents.PermissionRule(tools=("connector_sentry_d183.search_issues", "connector_github_app.get_file_contents"), decision="allow"),
        agents.PermissionRule(tools=("skill.*",), decision="allow"),  # the built-in skill tool is gated too
    ),
)
```

- Rules match the **stable tool ID** `<namespace>.<name>`, with case-sensitive globs. A rule with a bare name and no namespace is rejected. SDK tools live under `client.` (`client.my_tool`); built-ins under `file_system.`, `skill.`, `self.`, `agents.`.
- First matching rule wins; no match → `default`. A denied call never runs: the model reads `Tool call denied by permission policy.` and the turn goes on. The call still appears in the turn's `tool_calls`.
- `decision="ask"` opens an approval the caller must answer (`on_approval=` on `agent.run()` / a session handle); a workflow cannot answer approvals.
- Do not combine `permission_policy=` with `permission_mode="ask"` or `"deny"`.

## Hooks

```python
class AuditTrail(agents.Hook):
    async def pre_tool_call(self, ctx: agents.HookCallbackContext, call: agents.hook.PreToolCall):
        print(call.tool_id, call.tool_name, call.arguments, ctx.session.id)
        return None  # continue unchanged
        # return agents.skip("reason the model reads")  # stop the call
        # return agents.hook.PreToolCallHookContinue(effective_arguments={...})  # rewrite the arguments
```

- Six points: `pre_agent_turn`, `pre_llm_call`, `post_llm_call`, `pre_tool_call`, `post_tool_call`, `post_agent_turn`.
- A `Hook` is a Pydantic model: its attributes are per-session state and must be JSON fields with defaults. Keep hooks stateless unless you need state.
- Hooks run before the permission policy; guardrails (`guardrails=[...]`) run before hooks.

## Skills

A skill is a folder with a `SKILL.md`: YAML frontmatter (`name`, `description`), then the body. `name` is lowercase letters, digits and hyphens. The model sees only name and description in the system prompt and loads the body with the built-in `skill` tool (tool ID `skill.read`).

```
agent/skills/query-sentry/SKILL.md
```

## Structured output

`output_type=Model` on the `Agent` constrains every turn (also when served or deployed). On a session handle, `session.run(q, output_type=Model)` parses the reply into the model (it does not change what the model is told). `Completed.output` is then `Model | None | agents.InvalidOutput(raw, error)`.

## Run a turn and read the record

```python
session = await agent.start_session()
try:
    state = await session.run("question", output_type=InvestigationReport, timeout=600)
    match state:
        case agents.Completed(output=InvestigationReport() as report):
            ...
        case agents.Completed(output=agents.InvalidOutput(raw=raw, error=error)):
            ...
        case agents.Completed(stop_reason=reason):
            ...  # e.g. "iteration_limit", no final answer
        case agents.Failed(error=error):
            error.code, error.message, error.retryable
        case agents.Blocked():
            ...  # waits on an approval or client tool
        case agents.Running():
            ...  # timeout= passed first
        case agents.Busy():
            ...  # another turn is running on this session
        case agents.Busy():
            ...  # another turn is running on this session
finally:
    await session.cancel()  # ends the session and frees its sandbox
```

- Every record but `Busy` has `session_id`, `turn_id`, `text` (latest assistant text) and `tool_calls: tuple[agents.ToolCall, ...]` with `.name` (the tool ID, e.g. `connector_sentry_d183.search_issues`, `skill.read`) and `.arguments`.
- `agent.run(q, output_type=Model)` returns the same `TurnState` records for a one-shot turn; this repository uses `start_session()` + `session.run()` so the session ID is known before the turn starts and the session is cancelled afterwards.
- A served or deployed agent: `agents.Agent.load_from(agent_name="...")` then the same `start_session()` / `session.run()`. `MISTRAL_API_KEY` authenticates. `session.run(..., idempotency_key=key)` makes a retried call join the running turn instead of starting another.

## Testing without a model

```python
model = agents.testing.ScriptedModel(
    [
        agents.testing.call("skill", name="query-sentry"),  # the tool really runs, hooks and policy included
        agents.testing.reply('{"summary": "...", ...}'),  # text answer; ends the turn
    ]
)
session = await agent.override(model=model).start_session()  # override() copies the agent with a scripted model
state = await session.run("question")
model.assert_consumed()  # every step was used
```

- One step per model call, in order. A run that needs more steps than scripted fails with `completion_failed_permanently`.
- Scripted calls use the short tool name (`skill`, `my_tool`); the record shows the ID (`skill.read`, `client.my_tool`).
- Only tools with `model_access="direct"` (connector tools, `@agents.tool(model_access="direct")`) go through a scripted `call(...)`. Connector tools cannot run offline: build a connector-less twin of the agent for tests.
- `agents serve` and `agents deploy` refuse an overridden agent: override only in tests.

## Custom tools (for reference; this agent has none of its own)

```python
@agents.tool(model_access="direct")
async def search_issues(query: str, limit: int = 10) -> list[str]:
    """One-line description the model reads."""
```

Every parameter needs a type annotation; a docstring or `description=` is required. The tool ID is `client.<name>`.

## CLI (run from `agent/`)

```
uv run --env-file ../.env agents info .                      # what loads: model, connectors, skills, hooks, policy
uv run --env-file ../.env agents chat                        # in-process chat; -m "..." for one message
uv run --env-file ../.env agents serve . --name <name>       # durable local worker; sessions survive restarts
uv run --env-file ../.env agents deploy . --name <name>      # managed worker, built from the pushed git commit
uv run agents chat <name>                                    # chat with a served or deployed agent by name
uv run agents logs <name> --follow
```

`agents serve`/`deploy` need workflows enabled on the API key. `agents deploy` needs the project on GitHub with the Mistral GitHub app, and binds secrets with `--secret-binding VAR=secret:workspace:NAME`.

## Workflows (`mistralai-workflows`, inside the `mistral init --preset workflow` scaffold)

The scaffold's own skill, `workflow/.agents/skills/capability-workflows/SKILL.md`, is the authority on the SDK; this is
the shape this repository uses. One module per workflow in `workflow/apps/worker/src/worker/workflows/`; the scaffold's
`worker.py` entrypoint discovers every `@workflows.workflow.define` class in that package and serves it, so nothing is
registered by hand and the entrypoints are never edited.

```python
import mistralai.workflows as workflows


@workflows.activity(start_to_close_timeout=timedelta(minutes=11), retry_policy_max_attempts=1)
async def run_investigation(question: str, idempotency_key: str) -> dict:
    import mistralai.agents as agents  # inside the activity: a module-level import fails the determinism sandbox

    ...


@workflows.workflow.define(name="investigate", workflow_display_name="Investigate an error", workflow_description="...")
class Investigate:
    @workflows.workflow.entrypoint
    async def run(self, input: InvestigateInput) -> InvestigateResult:
        return await run_investigation(input.question, workflows.get_execution_id())
```

Inputs and outputs are Pydantic models. Everything that touches the network goes in an activity, and the libraries it
needs are imported there. `workflows.get_execution_id()` is stable across replays, which makes it a good idempotency key.
`workflows.discover_all_workflows_in_package("worker.workflows")` is what the tests and the worker use to find the class.
A dependency goes in through the scaffold's wrapper: `cd workflow && bash tools/uv.sh add --package workflows-app <pkg>`.
`cd workflow && mistral dev` runs the worker (reads `workflow/.env`, registers on its `DEPLOYMENT_NAME`, reloads on save).
