"""Tests for CoworkEnv."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.envs.experimental.cowork_env import (
    CoworkEnv,
    CoworkMonitorRubric,
    _safe_rel,
)


@pytest.fixture()
def sample_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Draft a memo"}],
                "answer": "x",
                "info": {},
            }
        ]
    )


@pytest.fixture()
def env(sample_dataset: Dataset) -> CoworkEnv:
    cowork_env = CoworkEnv(
        dataset=sample_dataset,
        rubric=vf.Rubric(),
        max_retries=1,
        base_delay=0.01,
    )
    cowork_env.logger = MagicMock()
    cowork_env.sandbox_client = MagicMock()
    return cowork_env


class TestSafeRel:
    def test_valid_relative_path(self) -> None:
        assert _safe_rel("foo/bar.txt").as_posix() == "foo/bar.txt"

    def test_rejects_traversal(self) -> None:
        with pytest.raises(ValueError):
            _safe_rel("../../etc/passwd")


class TestCoworkMonitorRubric:
    def test_has_expected_metrics(self) -> None:
        rubric = CoworkMonitorRubric()
        names = {func.__name__ for func in rubric._get_reward_funcs()}
        assert "total_tool_calls" in names
        assert "bash_calls" in names
        assert "artifact_count" in names
        assert "task_finished" in names


class TestCoworkEnv:
    def test_init_builds_run_command(self, env: CoworkEnv) -> None:
        assert env.remote_runner_path in env.run_command
        assert env.remote_input_path in env.run_command
        assert env.remote_state_path in env.run_command
        assert env.remote_logs_path in env.run_command

    def test_custom_system_prompt(self, sample_dataset: Dataset) -> None:
        env = CoworkEnv(
            dataset=sample_dataset,
            rubric=vf.Rubric(),
            system_prompt="Custom prompt",
        )
        assert env.system_prompt == "Custom prompt"

    def test_build_prompt_messages_strips_system(self, env: CoworkEnv) -> None:
        state = {
            "prompt": [
                {"role": "system", "content": env.system_prompt},
                {"role": "user", "content": "Draft a memo"},
            ]
        }
        assert env.build_prompt_messages(state) == [
            {"role": "user", "content": "Draft a memo"}
        ]

    def test_build_input_payload(self, env: CoworkEnv) -> None:
        state = {
            "prompt": [{"role": "user", "content": "Draft a memo"}],
            "info": {"user_answers": {"q1": "yes"}},
            "example_id": 7,
        }
        payload = env.build_input_payload(state)
        assert payload["workspace_root"] == "/workspace"
        assert payload["max_turns"] == env.max_turns
        assert payload["example_id"] == 7
        assert payload["messages"] == [{"role": "user", "content": "Draft a memo"}]

    @pytest.mark.asyncio
    async def test_post_sandbox_setup_uploads_runner_inputs_and_files(
        self, env: CoworkEnv
    ) -> None:
        env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(stdout="", stderr="", exit_code=0)
        )
        env.sandbox_client.upload_file = AsyncMock()
        env.upload_content = AsyncMock()
        env.upload_bundle = AsyncMock()
        state = {
            "sandbox_id": "sb-1",
            "prompt": [{"role": "user", "content": "Draft a memo"}],
            "info": {
                "initial_files": {"brief.txt": "hello"},
                "artifact_paths": ["final.md"],
            },
            "example_id": 0,
        }

        await env.post_sandbox_setup(state)

        assert state["workspace_root"] == "/workspace"
        assert state["working_dir"] == "/workspace"
        assert state["artifact_paths"] == ["final.md"]
        env.sandbox_client.execute_command.assert_awaited_once_with(
            "sb-1", "mkdir -p /cowork /workspace", working_dir=None
        )
        uploaded_paths = [call.args[2] for call in env.upload_content.await_args_list]
        assert env.remote_input_path in uploaded_paths
        assert env.remote_system_prompt_path in uploaded_paths
        env.sandbox_client.upload_file.assert_awaited_once_with(
            "sb-1",
            env.remote_runner_path,
            str(env.local_runner_path),
        )
        assert env.local_runner_path.name == "cowork_runner.py"
        assert env.local_runner_path == Path(
            "/root/verifiers/verifiers/envs/experimental/cowork_runner.py"
        )
        env.upload_bundle.assert_awaited_once_with(
            "sb-1", {"brief.txt": "hello"}, "/workspace"
        )

    @pytest.mark.asyncio
    async def test_build_env_vars(self, env: CoworkEnv) -> None:
        state = {
            "interception_base_url": "https://test.example/v1",
            "model": "gpt-4.1-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["OPENAI_BASE_URL"] == "https://test.example/v1"
        assert env_vars["OPENAI_MODEL"] == "gpt-4.1-mini"
        assert env_vars["PYTHONUNBUFFERED"] == "1"
        assert env_vars["COWORK_WORKSPACE_ROOT"] == "/workspace"

    @pytest.mark.asyncio
    async def test_post_rollout_collects_logs_runner_state_and_artifacts(
        self, env: CoworkEnv
    ) -> None:
        state = {
            "sandbox_id": "sb-1",
            "trajectory": [],
            "timing": {"total_ms": 0},
            "example_id": 0,
            "agent_exit_code": 0,
            "info": {"artifact_paths": ["final.md"]},
        }
        with patch.object(CliAgentEnv, "post_rollout", new=AsyncMock()) as super_post:
            env.read_file = AsyncMock(
                side_effect=[
                    "agent logs",
                    json.dumps(
                        {
                            "task_finished": True,
                            "task_summary": "done",
                            "last_plan": "read then write",
                            "todos": [{"id": "1", "content": "x", "status": "completed"}],
                            "memories": {"notes": {"title": "Notes", "body": "hello"}},
                        }
                    ),
                    "final deliverable",
                ]
            )

            await env.post_rollout(state)

        assert state["agent_logs"] == "agent logs"
        assert state["task_finished"] is True
        assert state["task_summary"] == "done"
        assert state["last_plan"] == "read then write"
        assert state["artifacts"] == {"final.md": "final deliverable"}
        super_post.assert_awaited_once()


class TestLazyExport:
    def test_vf_cowork_env(self) -> None:
        assert vf.CoworkEnv is CoworkEnv
