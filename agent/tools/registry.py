"""Keeps track of available tools and dispatches model-requested calls."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def schemas(self) -> List[dict]:
        """Ollama-compatible tool schemas for the /api/chat 'tools' field."""
        return [tool.to_ollama_schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: Any) -> ToolResult:
        """Run a tool by name. Never trusts the caller: unknown tools and
        bad arguments both come back as a failed ToolResult rather than
        raising, since the caller is a model-driven request.
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            return ToolResult(ok=False, output=f"Unknown tool '{name}'. Available tools: {available}")
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        return tool.execute(raw_arguments)
