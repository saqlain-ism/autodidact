"""Make MCP tool schemas portable across model providers.

MCP servers generate JSON Schema from Python type hints. That schema uses
``$ref``/``$defs``, ``anyOf`` with ``null`` for optional values, ``title`` on
every field, and ``default`` values. OpenAI-compatible endpoints differ in how
much of that they accept (Gemini's compatibility layer is the strictest), so
we reduce every schema to a plain, widely supported subset:

* ``$ref`` is inlined and ``$defs`` dropped;
* ``anyOf: [X, null]`` becomes ``X`` (the field is optional anyway);
* ``title`` is dropped;
* ``default`` is moved into the description, so the model still sees it.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def sanitize(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})
    return _clean(copy.deepcopy(schema), defs)


def _clean(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_clean(x, defs) for x in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
        return _clean(merged, defs)
    if "anyOf" in node:
        options = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(options) == 1:
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            return _clean({**options[0], **rest}, defs)
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("title", "$defs"):
            continue
        if key == "default":
            continue
        out[key] = _clean(value, defs)
    if "default" in node and node["default"] is not None:
        hint = f"Default: {json.dumps(node['default'])}."
        out["description"] = f"{out['description']} {hint}" if out.get("description") else hint
    return out


def to_openai_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP tool specs (name, description, input_schema) to OpenAI ``tools``."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description") or "",
                "parameters": sanitize(
                    s.get("input_schema") or {"type": "object", "properties": {}}
                ),
            },
        }
        for s in specs
    ]
