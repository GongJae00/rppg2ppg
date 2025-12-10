from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in {"null", "none"}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


def load_toon(path: str | Path) -> dict:
    """Minimal TOON (YAML-like) parser for the subset we need."""
    lines = Path(path).read_text().splitlines()
    filtered: list[tuple[int, str]] = []
    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        filtered.append((indent, line.strip()))

    idx = 0

    def parse_block(expected_indent: int) -> Any:
        nonlocal idx
        items = []
        mapping = {}
        is_list = False

        while idx < len(filtered):
            indent, text = filtered[idx]
            if indent < expected_indent:
                break
            if indent > expected_indent:
                raise ValueError(f"Unexpected indent at line: {text}")

            if text.startswith("- "):
                is_list = True
                value_part = text[2:]
                idx += 1
                if ":" in value_part:
                    key, val = value_part.split(":", 1)
                    val = val.strip()
                    if val == "":
                        value = parse_block(expected_indent + 2)
                    else:
                        value = {key.strip(): _parse_scalar(val)}
                elif value_part == "":
                    value = parse_block(expected_indent + 2)
                else:
                    value = _parse_scalar(value_part)
                items.append(value)
            else:
                if is_list:
                    raise ValueError("Mixed list and mapping at same level")
                if ":" not in text:
                    raise ValueError(f"Invalid line: {text}")
                key, val = text.split(":", 1)
                key = key.strip()
                val = val.strip()
                idx += 1
                if val == "":
                    mapping[key] = parse_block(expected_indent + 2)
                else:
                    mapping[key] = _parse_scalar(val)

        return items if is_list else mapping

    parsed = parse_block(0)
    return parsed
