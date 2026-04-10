#!/usr/bin/env python3
"""Cowork runner executed inside sandboxed CliAgentEnv rollouts."""

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--state-file", required=True)
    return parser.parse_args()


def safe_rel(file_path: str) -> PurePosixPath:
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


def workspace_path(root: Path, file_path: str) -> Path:
    rel = safe_rel(file_path)
    return root if rel == PurePosixPath(".") else root / rel


def tool_schemas() -> list[dict[str, Any]]:
    question_item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "prompt": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["id", "prompt"],
        "additionalProperties": False,
    }
    todo_item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "content": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["id", "content", "status"],
        "additionalProperties": False,
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file under the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write text content to a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace text inside an existing workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob_files",
                "description": "List files under the workspace matching a glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_search",
                "description": "Search file contents under the workspace using a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "max_matches": {"type": "integer"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "todo_write",
                "description": "Update the session todo list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "merge": {"type": "boolean"},
                        "todos": {
                            "type": "array",
                            "items": todo_item,
                        },
                    },
                    "required": ["merge", "todos"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory",
                "description": "Manage scratch memories for this rollout.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "memory_id": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "text": {"type": "string"},
                        "offset": {"type": "integer"},
                        "new_memory_id": {"type": "string"},
                        "new_title": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user_question",
                "description": "Return scripted clarification answers from task metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": question_item,
                        },
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exit_plan_mode",
                "description": "Record a short plan for traceability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string"},
                    },
                    "required": ["plan"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_task",
                "description": "Signal task completion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
    ]


class CoworkRuntime:
    def __init__(self, payload: dict[str, Any]):
        self.workspace_root = Path(payload["workspace_root"]).resolve()
        self.info = payload.get("info") or {}
        self.max_turns = int(payload.get("max_turns", 32))
        self.messages = list(payload.get("messages") or [])
        self.memories: dict[str, dict[str, str]] = {}
        self.todos: list[dict[str, str]] = []
        self.task_finished = False
        self.task_summary: str | None = None
        self.last_plan: str | None = None
        self.tool_call_counts: Counter[str] = Counter()

    def read_file(
        self, file_path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        path = workspace_path(self.workspace_root, file_path)
        if not path.is_file():
            return f"Error: not a file: {file_path}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if offset is not None:
            lines = lines[max(offset - 1, 0) :]
        if limit is not None:
            lines = lines[:limit]
        out = "\n".join(lines)
        return out if out else "(empty)"

    def write_file(self, file_path: str, content: str) -> str:
        path = workspace_path(self.workspace_root, file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {file_path}"

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        path = workspace_path(self.workspace_root, file_path)
        if not path.is_file():
            return f"Error: not a file: {file_path}"
        text = path.read_text(encoding="utf-8")
        if old_string not in text:
            return "Error: old_string not found in file."
        if replace_all:
            new_text = text.replace(old_string, new_string)
            count = text.count(old_string)
        else:
            new_text = text.replace(old_string, new_string, 1)
            count = 1
        path.write_text(new_text, encoding="utf-8")
        return f"Applied edit to {file_path} ({count} replacement(s))."

    def glob_files(self, pattern: str, path: str | None = None) -> str:
        base = (
            self.workspace_root
            if path is None
            else workspace_path(self.workspace_root, path)
        )
        if not base.exists():
            return f"Error: path not found: {path or '.'}"
        matches = []
        for fp in base.rglob("*"):
            if not fp.is_file():
                continue
            try:
                rel = fp.resolve().relative_to(self.workspace_root)
            except ValueError:
                continue
            if rel.match(pattern):
                matches.append(rel.as_posix())
        return "\n".join(sorted(set(matches))) if matches else "No matches."

    def grep_search(
        self, pattern: str, path: str | None = None, max_matches: int = 40
    ) -> str:
        base = (
            self.workspace_root
            if path is None
            else workspace_path(self.workspace_root, path)
        )
        if not base.exists():
            return f"Error: path not found: {path or '.'}"
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        lines = []
        count = 0
        for fp in base.rglob("*"):
            if not fp.is_file():
                continue
            try:
                rel = fp.resolve().relative_to(self.workspace_root)
            except ValueError:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    count += 1
                    if len(lines) < max_matches:
                        lines.append(f"{rel.as_posix()}:{line_no}:{line[:500]}")
                    if count >= max_matches:
                        break
            if count >= max_matches:
                break
        if not lines:
            return "No matches."
        suffix = (
            f"\n... truncated after {max_matches} matches"
            if count >= max_matches
            else ""
        )
        return "\n".join(lines) + suffix

    def bash(self, command: str, timeout: int = 30) -> str:
        try:
            proc = subprocess.run(
                command,
                cwd=self.workspace_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        output = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        combined = output
        if stderr:
            combined = (
                f"{combined}\nstderr:\n{stderr}" if combined else f"stderr:\n{stderr}"
            )
        prefix = f"[exit code: {proc.returncode}]\n" if proc.returncode != 0 else ""
        return prefix + (combined or "(no output)")

    def todo_write(self, merge: bool, todos: list[dict[str, Any]]) -> str:
        if not isinstance(todos, list):
            return "Error: todos must be a list."
        if not merge:
            self.todos = []
        by_id = {
            item.get("id"): item for item in self.todos if isinstance(item, dict)
        }
        for item in todos:
            if not isinstance(item, dict) or "id" not in item:
                continue
            by_id[item["id"]] = {
                "id": item["id"],
                "content": item.get("content", ""),
                "status": item.get("status", "pending"),
            }
        self.todos = list(by_id.values())
        return f"Todo list updated ({len(self.todos)} items)."

    def memory(
        self,
        command: str,
        memory_id: str | None = None,
        title: str | None = None,
        content: str | None = None,
        text: str | None = None,
        offset: int | None = None,
        new_memory_id: str | None = None,
        new_title: str | None = None,
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
    ) -> str:
        cmd = command.strip().lower()
        if cmd == "view":
            if not memory_id:
                if not self.memories:
                    return "No memories."
                return json.dumps(
                    {
                        key: {"title": value["title"], "len": len(value["body"])}
                        for key, value in self.memories.items()
                    },
                    indent=2,
                )
            if memory_id not in self.memories:
                return f"Error: unknown memory_id {memory_id!r}"
            record = self.memories[memory_id]
            return json.dumps(
                {"id": memory_id, "title": record["title"], "body": record["body"]},
                indent=2,
            )
        if cmd == "create":
            if not memory_id:
                return "Error: memory_id required for create."
            if memory_id in self.memories:
                return f"Error: memory {memory_id!r} already exists."
            self.memories[memory_id] = {
                "title": title or memory_id,
                "body": content or "",
            }
            return f"Created memory {memory_id!r}."
        if cmd == "insert":
            if not memory_id or memory_id not in self.memories:
                return "Error: valid memory_id required for insert."
            body = self.memories[memory_id]["body"]
            insertion = text or ""
            if offset is None or offset >= len(body):
                self.memories[memory_id]["body"] = body + insertion
            else:
                self.memories[memory_id]["body"] = (
                    body[:offset] + insertion + body[offset:]
                )
            return f"Inserted into {memory_id!r}."
        if cmd == "delete":
            if not memory_id or memory_id not in self.memories:
                return "Error: valid memory_id required for delete."
            del self.memories[memory_id]
            return f"Deleted {memory_id!r}."
        if cmd == "rename":
            if not memory_id or memory_id not in self.memories:
                return "Error: valid memory_id required for rename."
            current_id = memory_id
            if new_memory_id and new_memory_id != current_id:
                if new_memory_id in self.memories:
                    return f"Error: target id {new_memory_id!r} exists."
                self.memories[new_memory_id] = self.memories.pop(current_id)
                current_id = new_memory_id
            if new_title is not None:
                self.memories[current_id]["title"] = new_title
            return f"Renamed/updated {current_id!r}."
        if cmd == "str_replace":
            if not memory_id or memory_id not in self.memories:
                return "Error: valid memory_id required for str_replace."
            if old_string is None or new_string is None:
                return "Error: old_string and new_string required."
            body = self.memories[memory_id]["body"]
            if old_string not in body:
                return "Error: old_string not found in memory body."
            self.memories[memory_id]["body"] = (
                body.replace(old_string, new_string)
                if replace_all
                else body.replace(old_string, new_string, 1)
            )
            return f"Updated body of {memory_id!r}."
        return f"Error: unknown command {command!r}"

    def ask_user_question(self, questions: list[dict[str, Any]]) -> str:
        answers_raw = self.info.get("user_answers") or {}
        answers = answers_raw if isinstance(answers_raw, dict) else {}
        out_lines = []
        for question in questions:
            if not isinstance(question, dict):
                continue
            qid = str(question.get("id", ""))
            prompt = question.get("prompt", "")
            options = question.get("options")
            if qid in answers:
                out_lines.append(f"[{qid}] {prompt}\n→ {answers[qid]}")
            else:
                hint = f" Options: {options}" if options else ""
                out_lines.append(f"[{qid}] {prompt}\n→ (no scripted answer){hint}")
        return "\n\n".join(out_lines) if out_lines else "No questions parsed."

    def exit_plan_mode(self, plan: str) -> str:
        self.last_plan = plan
        return "Plan recorded. Proceed with the task using tools as needed."

    def finish_task(self, summary: str) -> str:
        self.task_finished = True
        self.task_summary = summary
        return f"Task completed: {summary}"

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.tool_call_counts[name] += 1
        handler = getattr(self, name, None)
        if handler is None:
            return f"Error: unknown tool {name!r}"
        try:
            return str(handler(**arguments))
        except Exception as exc:
            return f"Error: {exc}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_finished": self.task_finished,
            "task_summary": self.task_summary,
            "last_plan": self.last_plan,
            "todos": self.todos,
            "memories": self.memories,
            "tool_call_counts": dict(self.tool_call_counts),
        }


def load_input(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    system_prompt = None
    system_path = Path(args.system_prompt_file)
    if system_path.exists():
        system_prompt = system_path.read_text(encoding="utf-8").strip() or None
    return payload, system_prompt


def append_assistant_message(messages: list[dict[str, Any]], message: Any) -> None:
    assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        assistant["tool_calls"] = [
            tool_call.model_dump(mode="json") for tool_call in message.tool_calls
        ]
    messages.append(assistant)


def main() -> int:
    args = parse_args()
    payload, system_prompt = load_input(args)
    runtime = CoworkRuntime(payload)
    messages = list(runtime.messages)
    if system_prompt and (not messages or messages[0].get("role") != "system"):
        messages = [{"role": "system", "content": system_prompt}] + messages

    client = OpenAI(
        api_key="intercepted",
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=float(os.environ.get("OPENAI_TIMEOUT", "3600")),
    )

    tool_defs = tool_schemas()
    state_path = Path(args.state_file)

    def persist_state() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(runtime.snapshot(), indent=2), encoding="utf-8")

    try:
        for _ in range(runtime.max_turns):
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL"],
                messages=messages,
                tools=tool_defs,
            )
            message = response.choices[0].message
            append_assistant_message(messages, message)
            if not message.tool_calls:
                break
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    tool_result = f"Error: invalid JSON arguments: {exc}"
                else:
                    if not isinstance(arguments, dict):
                        tool_result = "Error: tool arguments must be a JSON object."
                    else:
                        tool_result = runtime.call_tool(
                            tool_call.function.name, arguments
                        )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
                if runtime.task_finished:
                    break
            if runtime.task_finished:
                break
    finally:
        persist_state()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
