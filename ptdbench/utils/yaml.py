"""给 metadata 使用的轻量 YAML 读取器。

这里只支持 PTDBench schema 实际需要的 YAML 子集，不引入额外依赖，也不
试图成为完整的 YAML 解析器。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


class SimpleYamlError(ValueError):
    """轻量 YAML 解析失败。"""


@dataclass(frozen=True)
class _Line:
    number: int
    indent: int
    content: str


def load_simple_yaml_text(text: str) -> Any:
    """解析 mapping、list、嵌套缩进和基础标量类型。"""
    lines = _tokenize(text.splitlines())
    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0].indent)
    if index != len(lines):
        line = lines[index]
        raise SimpleYamlError(f"unexpected content at line {line.number}")
    return value


def load_simple_yaml(path: str) -> Any:
    """读取并解析一个 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as file:
        return load_simple_yaml_text(file.read())


def _tokenize(raw_lines: Sequence[str]) -> List[_Line]:
    tokens: List[_Line] = []
    for index, raw_line in enumerate(raw_lines, 1):
        stripped = _strip_comment(raw_line.rstrip("\n"))
        if not stripped.strip():
            continue
        if "\t" in stripped:
            raise SimpleYamlError(f"tabs are not supported (line {index})")
        indent = len(stripped) - len(stripped.lstrip(" "))
        tokens.append(_Line(index, indent, stripped[indent:]))
    return tokens


def _strip_comment(line: str) -> str:
    # 只有不在引号内的 # 才是注释起点。
    in_single = False
    in_double = False
    escaped = False
    for position, char in enumerate(line):
        if char == "\\" and in_double and not escaped:
            escaped = True
            continue
        if char == "'" and not in_double and not escaped:
            in_single = not in_single
        elif char == '"' and not in_single and not escaped:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:position].rstrip()
        escaped = False
    return line.rstrip()


def _parse_block(lines: Sequence[_Line], index: int, indent: int) -> Tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    if lines[index].indent < indent:
        return {}, index
    if lines[index].content.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: Sequence[_Line], index: int, indent: int) -> Tuple[dict, int]:
    result = {}
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise SimpleYamlError(f"unexpected indentation at line {line.number}")
        if line.content.startswith("- "):
            break
        key, value, has_value = _split_key_value(line.content, line.number)
        index += 1
        if has_value:
            result[key] = _parse_scalar(value, line.number)
            continue
        if index < len(lines) and lines[index].indent > indent:
            child, index = _parse_block(lines, index, lines[index].indent)
            result[key] = child
        else:
            result[key] = None
    return result, index


def _parse_list(lines: Sequence[_Line], index: int, indent: int) -> Tuple[list, int]:
    result = []
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise SimpleYamlError(f"unexpected indentation at line {line.number}")
        if not line.content.startswith("- "):
            break
        payload = line.content[2:].strip()
        index += 1
        if not payload:
            if index < len(lines) and lines[index].indent > indent:
                item, index = _parse_block(lines, index, lines[index].indent)
            else:
                item = None
            result.append(item)
            continue
        # Flow collections are checked before the `key: value` split below.
        # Otherwise `- {path: a, sha256: b}` splits at its first colon and
        # yields a key literally named "{path" holding the rest as a string --
        # silently, which for a checksum manifest means the sha256 key is simply
        # absent and verification is skipped. _parse_scalar rejects any flow
        # form it cannot evaluate outright.
        if payload[0] in "{[":
            result.append(_parse_scalar(payload, line.number))
            if index < len(lines) and lines[index].indent > indent:
                raise SimpleYamlError(
                    f"flow list item at line {line.number} cannot have a nested block"
                )
            continue
        if ":" in payload:
            key, value, has_value = _split_key_value(payload, line.number)
            item = {key: _parse_scalar(value, line.number) if has_value else None}
            if not has_value and index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(lines, index, lines[index].indent)
                item[key] = child
            elif has_value and index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(lines, index, lines[index].indent)
                if not isinstance(child, dict):
                    raise SimpleYamlError(
                        f"list item at line {line.number} expects mapping continuation"
                    )
                item.update(child)
            result.append(item)
            continue
        result.append(_parse_scalar(payload, line.number))
        if index < len(lines) and lines[index].indent > indent:
            raise SimpleYamlError(
                f"scalar list item at line {line.number} cannot have a nested block"
            )
    return result, index


def _split_key_value(content: str, line_number: int) -> Tuple[str, str, bool]:
    # 按第一个不在引号内的冒号切分 key/value。
    in_single = False
    in_double = False
    for position, char in enumerate(content):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ":" and not in_single and not in_double:
            key = content[:position].strip()
            value = content[position + 1 :].strip()
            if not key:
                raise SimpleYamlError(f"missing key before ':' at line {line_number}")
            return key, value, bool(value)
    raise SimpleYamlError(f"expected key: value at line {line_number}")


def _parse_scalar(raw: str, line_number: int | None = None) -> Any:
    text = raw.strip()
    if text in {"", "~", "null", "Null", "NULL"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text and text[0] in "{[('\"":
        try:
            return ast.literal_eval(text)
        except Exception:
            pass
        if text[0] in "{[":
            # A flow collection we cannot evaluate, e.g. `{path: a, sha256: b}`
            # (bare words are not Python literals). Falling through would return
            # it as a plain string and the caller would see a mapping with one
            # nonsense key, so refuse instead -- a mangled checksum manifest is
            # worse than a parse error.
            where = f" at line {line_number}" if line_number is not None else ""
            raise SimpleYamlError(
                f"unsupported flow collection{where}: {text!r}; "
                "write it in block style (one `key: value` per line)"
            )
    if _looks_int(text):
        try:
            return int(text)
        except Exception:
            pass
    if _looks_float(text):
        try:
            return float(text)
        except Exception:
            pass
    return text


def _looks_int(text: str) -> bool:
    if text.startswith(("+", "-")):
        text = text[1:]
    return text.isdigit()


def _looks_float(text: str) -> bool:
    if text.count(".") != 1:
        return False
    left, right = text.split(".", 1)
    if left.startswith(("+", "-")):
        left = left[1:]
    return left.isdigit() and right.isdigit()
