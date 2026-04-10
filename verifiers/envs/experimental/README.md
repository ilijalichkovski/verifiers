# Experimental Environments

Newer and more experimental environment classes that may have some sharper edges + change more frequently.

## GymEnv

Universal runner for Gym-compatible environments. Wraps any environment that implements `reset(seed)` and `step(action)` methods (following the OpenAI Gym / Gymnasium API). Supports both old-style 4-tuple and new-style 5-tuple step returns.

## MCPEnv

Environment for integrating MCP (Model Context Protocol) servers as tools. Connects to one or more MCP servers via stdio transport and exposes their tools to the model. Useful for giving models access to external services like web search, file fetching, or any MCP-compatible tool server.

## CliAgentEnv

Environment for running custom agent code inside sandboxes. Intercepts the agent's OpenAI API requests via an HTTP proxy server, with each request triggering one `MultiTurnEnv` rollout step.

## HarborEnv

`CliAgentEnv` subclass that loads Harbor-format tasks. Harbor is a task format for agent benchmarks with structured task directories containing `task.toml` configuration and `instruction.md` prompts, along with test scripts for computing rewards.

## RLMEnv

Environment implementing Recursive Language Models (RLMs).

## CoworkEnv

`CliAgentEnv` subclass for knowledge-work agents (documents, policies, deliverables). Uploads a lightweight Cowork runner into the sandbox, intercepts its OpenAI-compatible requests, exposes a non-coding tool surface (workspace files, bash, todos, scratch memory, scripted clarifications, plan logging, `finish_task`), and captures logs plus declared artifacts for scoring.
