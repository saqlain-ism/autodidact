"""One episode against a sandbox app, spoken over MCP.

Everything that touches an app (the agent, reference solutions, tests) goes
through ``ToolSession``. So if a task is solvable here, it is solvable using
only what the MCP server exposes; nothing reaches into the database behind the
agent's back. Checkers are the one exception: they read ``session.app`` to
inspect the final state, like an auditor would.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from mcp import Client

from autodidact.envs.inventory.app import InventoryApp
from autodidact.envs.inventory.server import build_server

# Tool errors are normal during exploration; the SDK logs each one at ERROR level.
logging.getLogger("mcp").setLevel(logging.CRITICAL)


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None

    def text(self) -> str:
        """What the agent sees."""
        if self.ok:
            return json.dumps(self.data, ensure_ascii=False)
        return f"ERROR: {self.error}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: ToolResult


@dataclass
class ToolSession:
    """Async context manager wrapping one fresh world and an MCP client to it."""

    seed: int
    app: InventoryApp = field(init=False)
    calls: list[ToolCall] = field(init=False, default_factory=list)

    async def __aenter__(self) -> ToolSession:
        self.app = InventoryApp.from_seed(self.seed)
        self._client = Client(build_server(self.app))
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.__aexit__(*exc)

    async def tool_specs(self) -> list[dict[str, Any]]:
        """Tool names, descriptions and JSON schemas, as the server advertises them."""
        listed = await self._client.list_tools()
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in listed.tools
        ]

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = arguments or {}
        raw = await self._client.call_tool(name, arguments)
        text = "".join(getattr(c, "text", "") for c in raw.content)
        if raw.is_error:
            text = re.sub(r"^Error executing tool \w+: ", "", text.strip())
            result = ToolResult(ok=False, error=text)
        else:
            try:
                result = ToolResult(ok=True, data=json.loads(text))
            except json.JSONDecodeError:
                result = ToolResult(ok=True, data=text)
        self.calls.append(ToolCall(name, arguments, result))
        return result

    async def must(self, tool: str, /, **arguments: Any) -> Any:
        """Call a tool and return its data, raising on error. For scripted code only."""
        result = await self.call(tool, arguments)
        if not result.ok:
            raise RuntimeError(f"{tool}({arguments}) failed: {result.error}")
        return result.data

    async def collect(self, tool: str, /, **arguments: Any) -> list[dict]:
        """Follow ``next_cursor`` to the end of a paginated list."""
        items, cursor = [], None
        while True:
            page = await self.must(tool, cursor=cursor, page_size=20, **arguments)
            items.extend(page["items"])
            cursor = page.get("next_cursor")
            if not cursor:
                return items
