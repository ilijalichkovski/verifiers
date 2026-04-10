"""CoworkEnv — CliAgentEnv for knowledge-work agents."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.types import AssistantMessage, Messages, ToolCall
from verifiers.utils.logging_utils import truncate
from verifiers.utils.message_utils import normalize_messages

logger = logging.getLogger(__name__)

COWORK_TOOLS = [
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "glob_files",
    "grep_search",
    "todo_write",
    "memory",
    "ask_user_question",
    "exit_plan_mode",
    "finish_task",
]

DEFAULT_SYSTEM_PROMPT = """\
You are Cowork, a careful knowledge-work agent for documents, policies, research notes, and deliverables.

Work only inside the provided sandbox workspace.
Read source material before writing summaries or final outputs.
Use the available tools to inspect files, maintain scratch memory, track todos, ask scripted clarification questions, and finish the task explicitly when done.
Prefer precise, concise deliverables grounded in the provided materials."""

DEFAULT_INSTALL_COMMAND = (
    "python -m pip install --quiet --disable-pip-version-check openai"
)
DEFAULT_RUN_COMMAND_TEMPLATE = """\
set -eo pipefail

{install_command}

mkdir -p {asset_dir} {agent_workdir}
cd {agent_workdir}
python {runner_path} \
  --input-file {input_path} \
  --system-prompt-file {system_prompt_path} \
  --state-file {state_path} 2>&1 | tee {logs_path}
"""


def _parse_info(info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        try:
            parsed = json.loads(info)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _safe_rel(file_path: str) -> PurePosixPath:
    raw = file_path.strip()
    if not raw:
        raise ValueError("Path must not be empty.")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("Path traversal is not allowed.")
        parts.append(part)
    return PurePosixPath(*parts) if parts else PurePosixPath(".")


class CoworkMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks Cowork runner tool usage and outputs."""

    def __init__(self, tool_names: list[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.tool_names = list(tool_names or COWORK_TOOLS)
        self.add_metric(self.total_tool_calls)
        self.add_metric(self.unique_tools_used)
        self.add_metric(self.has_tool_calls)
        self.add_metric(self.artifact_count)
        self.add_metric(self.task_finished)
        for tool_name in self.tool_names:
            self.add_metric(self._make_tool_count_metric(tool_name))

    @staticmethod
    def _count_tool_calls(completion: Messages) -> Counter[str]:
        counts: Counter[str] = Counter()
        assert isinstance(completion, list)
        for msg in completion:
            if not isinstance(msg, AssistantMessage):
                continue
            tool_calls = msg.tool_calls
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if isinstance(tool_call, ToolCall):
                    counts[tool_call.name] += 1
        return counts

    async def total_tool_calls(self, completion: Messages) -> float:
        return float(sum(self._count_tool_calls(completion).values()))

    async def unique_tools_used(self, completion: Messages) -> float:
        return float(len(self._count_tool_calls(completion)))

    async def has_tool_calls(self, completion: Messages) -> float:
        return float(bool(self._count_tool_calls(completion)))

    def _make_tool_count_metric(self, tool_name: str) -> Callable:
        async def tool_count(completion: Messages) -> float:
            counts = self._count_tool_calls(completion)
            return float(counts.get(tool_name, 0))

        tool_count.__name__ = f"{tool_name}_calls"
        return tool_count

    async def artifact_count(self, state: vf.State) -> float:
        return float(len(state.get("artifacts", {})))

    async def task_finished(self, state: vf.State) -> float:
        return float(bool(state.get("task_finished")))


class CoworkEnv(CliAgentEnv):
    """Knowledge-work agent environment built on CliAgentEnv."""

    DEFAULT_AGENT_WORKDIR = "/workspace"
    DEFAULT_ASSET_DIR = "/cowork"
    DEFAULT_INSTALL_COMMAND = DEFAULT_INSTALL_COMMAND
    DEFAULT_RUN_COMMAND_TEMPLATE = DEFAULT_RUN_COMMAND_TEMPLATE
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Dataset | None = None,
        asset_dir: str = DEFAULT_ASSET_DIR,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        run_command_template: str = DEFAULT_RUN_COMMAND_TEMPLATE,
        artifact_paths: list[str] | None = None,
        docker_image: str = "python:3.11-slim",
        **kwargs: Any,
    ):
        self.asset_dir = asset_dir
        self.agent_workdir = agent_workdir
        self.default_artifact_paths = list(artifact_paths or [])

        run_command = self.build_run_command(
            run_command_template,
            install_command=install_command,
        )

        super().__init__(
            run_command=run_command,
            dataset=dataset,
            eval_dataset=eval_dataset if eval_dataset is not None else dataset,
            system_prompt=system_prompt,
            docker_image=docker_image,
            **kwargs,
        )
        self.add_rubric(CoworkMonitorRubric())

    @property
    def remote_runner_path(self) -> str:
        return f"{self.asset_dir}/runner.py"

    @property
    def local_runner_path(self) -> Path:
        return Path(__file__).with_name("cowork_runner.py")

    @property
    def remote_input_path(self) -> str:
        return f"{self.asset_dir}/input.json"

    @property
    def remote_state_path(self) -> str:
        return f"{self.asset_dir}/state.json"

    @property
    def remote_system_prompt_path(self) -> str:
        return f"{self.asset_dir}/system.txt"

    @property
    def remote_logs_path(self) -> str:
        return f"{self.asset_dir}/logs.txt"

    def build_prompt_messages(self, state: vf.State) -> list[dict[str, Any]]:
        prompt: list[dict[str, Any]] = []
        for message in normalize_messages(state.get("prompt", [])):
            if hasattr(message, "model_dump"):
                prompt.append(message.model_dump(mode="json"))
            elif isinstance(message, dict):
                prompt.append(dict(message))
            else:
                raise TypeError(
                    f"Unsupported prompt message type: {type(message).__name__}"
                )
        if self.system_prompt and prompt and prompt[0].get("role") == "system":
            return prompt[1:]
        return prompt

    def build_input_payload(self, state: vf.State) -> dict[str, Any]:
        return {
            "messages": self.build_prompt_messages(state),
            "info": _parse_info(state.get("info")),
            "workspace_root": self.agent_workdir,
            "max_turns": self.max_turns,
            "example_id": state.get("example_id"),
        }

    def _artifact_paths_for_state(self, state: vf.State) -> list[str]:
        info_paths = _parse_info(state.get("info")).get("artifact_paths")
        raw_paths = (
            info_paths if isinstance(info_paths, list) else self.default_artifact_paths
        )
        cleaned: list[str] = []
        for path in raw_paths:
            if not isinstance(path, str):
                continue
            cleaned.append(_safe_rel(path).as_posix())
        return cleaned

    def _initial_files_for_state(self, state: vf.State) -> dict[str, str]:
        raw = _parse_info(state.get("info")).get("initial_files")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for rel_path, content in raw.items():
            if not isinstance(rel_path, str):
                continue
            out[_safe_rel(rel_path).as_posix()] = str(content)
        return out

    async def post_sandbox_setup(self, state: vf.State) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        state["workspace_root"] = self.agent_workdir
        state["working_dir"] = self.agent_workdir
        state["artifact_paths"] = self._artifact_paths_for_state(state)

        dirs = [self.asset_dir, self.agent_workdir]
        await self.sandbox_client.execute_command(
            sandbox_id,
            f"mkdir -p {' '.join(dirs)}",
            working_dir=None,
        )

        await self.upload_content(
            sandbox_id,
            json.dumps(self.build_input_payload(state), indent=2),
            self.remote_input_path,
        )
        await self.sandbox_client.upload_file(
            sandbox_id,
            self.remote_runner_path,
            str(self.local_runner_path),
        )
        if self.system_prompt:
            await self.upload_content(
                sandbox_id,
                self.system_prompt,
                self.remote_system_prompt_path,
            )

        initial_files = self._initial_files_for_state(state)
        if initial_files:
            await self.upload_bundle(sandbox_id, initial_files, self.agent_workdir)

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        env_vars["PYTHONUNBUFFERED"] = "1"
        env_vars["COWORK_WORKSPACE_ROOT"] = self.agent_workdir
        return env_vars

    async def normalize_response(self, response: vf.Response) -> vf.Response:
        def _normalize() -> vf.Response:
            message = response.message
            normalized_tool_calls = message.tool_calls or []
            if message.tool_calls:
                normalized_tool_calls = []
                for tool_call in message.tool_calls:
                    if not isinstance(tool_call, ToolCall):
                        normalized_tool_calls.append(tool_call)
                        continue
                    try:
                        compact_arguments = json.dumps(
                            json.loads(tool_call.arguments),
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                    except (json.JSONDecodeError, TypeError):
                        compact_arguments = tool_call.arguments
                    normalized_tool_calls.append(
                        tool_call.model_copy(
                            update={
                                "name": tool_call.name.lower(),
                                "arguments": compact_arguments,
                            }
                        )
                    )
            normalized_message = message.model_copy(
                update={
                    "content": message.content or "",
                    "tool_calls": normalized_tool_calls,
                    "reasoning_content": message.reasoning_content or None,
                }
            )
            return response.model_copy(update={"message": normalized_message})

        return await asyncio.to_thread(_normalize)

    async def post_rollout(self, state: vf.State) -> None:
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                logs = await self.read_file(sandbox_id, self.remote_logs_path)
                state["agent_logs"] = logs or "<no logs>"
            except Exception as exc:
                logger.warning(f"Failed to collect Cowork logs: {exc}")

            try:
                runner_state_raw = await self.read_file(sandbox_id, self.remote_state_path)
                runner_state = (
                    json.loads(runner_state_raw)
                    if isinstance(runner_state_raw, str) and runner_state_raw
                    else {}
                )
            except Exception as exc:
                logger.warning(f"Failed to collect Cowork state: {exc}")
                runner_state = {}

            if isinstance(runner_state, dict):
                state["runner_state"] = runner_state
                state["task_finished"] = bool(runner_state.get("task_finished"))
                state["task_summary"] = runner_state.get("task_summary")
                state["last_plan"] = runner_state.get("last_plan")
                state["todos"] = runner_state.get("todos", [])
                state["memories"] = runner_state.get("memories", {})

            artifacts: dict[str, str] = {}
            for rel_path in state.get(
                "artifact_paths", self._artifact_paths_for_state(state)
            ):
                remote_path = f"{self.agent_workdir.rstrip('/')}/{rel_path}"
                content = await self.read_file(sandbox_id, remote_path)
                if content is not None:
                    artifacts[rel_path] = content
            state["artifacts"] = artifacts

            num_turns = len(state.get("trajectory", []))
            agent_error = state.get("agent_exit_code", 0) != 0
            agent_logs = state.get("agent_logs")
            if (agent_error or num_turns == 0) and agent_logs:
                logger.warning(
                    f"Cowork logs (example_id={state.get('example_id')}, "
                    f"exit_code={state.get('agent_exit_code')}, turns={num_turns}):\n"
                    f"{truncate(str(agent_logs), 4000)}"
                )

        await super().post_rollout(state)

    def build_run_command(
        self,
        run_command_template: str,
        install_command: str = DEFAULT_INSTALL_COMMAND,
    ) -> str:
        return run_command_template.format(
            install_command=install_command,
            asset_dir=self.asset_dir,
            agent_workdir=self.agent_workdir,
            runner_path=self.remote_runner_path,
            input_path=self.remote_input_path,
            system_prompt_path=self.remote_system_prompt_path,
            state_path=self.remote_state_path,
            logs_path=self.remote_logs_path,
        )
