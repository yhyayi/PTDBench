"""给本仓库受控 schema 使用的轻量 YAML 读取器。

它刻意保持小而无依赖，支持：
- 基于缩进的 mapping
- 使用 "- " 的 list
- 嵌套 mapping/list
- 行内空 dict/list：{} 和 []
- bool、null、int、float
- 引号字符串

它不是完整 YAML parser。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


class SimpleYamlError(ValueError):
    pass


@dataclass(frozen=True)
class _Line:
    number: int
    indent: int
    content: str


def load_simple_yaml_text(text: str) -> Any:
    # 解析 metadata 需要的 YAML 子集；目标是稳定可控，不追求完整 YAML 兼容。
    lines = _tokenize(text.splitlines())
    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0].indent)
    if index != len(lines):
        line = lines[index]
        raise SimpleYamlError(f"unexpected content at line {line.number}")
    return value


def load_simple_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return load_simple_yaml_text(f.read())


def _tokenize(raw_lines: Sequence[str]) -> List[_Line]:
    tokens: List[_Line] = []
    for idx, raw in enumerate(raw_lines, 1):
        stripped = _strip_comment(raw.rstrip("\n"))
        if not stripped.strip():
            continue
        if "\t" in stripped:
            raise SimpleYamlError(f"tabs are not supported (line {idx})")
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped[indent:]
        tokens.append(_Line(idx, indent, content))
    return tokens


def _strip_comment(line: str) -> str:
    # 去注释时要识别引号，避免把 "a # b" 这种字符串截断。
    in_single = False
    in_double = False
    escaped = False
    for pos, ch in enumerate(line):
        if ch == "\\" and in_double and not escaped:
            escaped = True
            continue
        if ch == "'" and not in_double and not escaped:
            in_single = not in_single
        elif ch == '"' and not in_single and not escaped:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:pos].rstrip()
        escaped = False
    return line.rstrip()


def _parse_block(lines: Sequence[_Line], index: int, indent: int) -> Tuple[Any, int]:
    # 根据当前块第一个有效 token 判断是 mapping 还是 list。
    if index >= len(lines):
        return {}, index
    line = lines[index]
    if line.indent < indent:
        return {}, index
    if line.content.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: Sequence[_Line], index: int, indent: int) -> Tuple[dict, int]:
    # 解析器按缩进工作：更深一层缩进代表一个嵌套块。
    out = {}
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
            out[key] = _parse_scalar(value)
            continue
        if index < len(lines) and lines[index].indent > indent:
            child, index = _parse_block(lines, index, lines[index].indent)
            out[key] = child
        else:
            out[key] = None
    return out, index


def _parse_list(lines: Sequence[_Line], index: int, indent: int) -> Tuple[list, int]:
    # list 元素可以是标量，也可以是行内 mapping。
    out = []
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
            out.append(item)
            continue
        if ":" in payload:
            key, value, has_value = _split_key_value(payload, line.number)
            item = {key: _parse_scalar(value) if has_value else None}
            if not has_value and index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(lines, index, lines[index].indent)
                item[key] = child
            elif has_value and index < len(lines) and lines[index].indent > indent:
                child, index = _parse_block(lines, index, lines[index].indent)
                if not isinstance(child, dict):
                    raise SimpleYamlError(f"list item at line {line.number} expects mapping continuation")
                item.update(child)
            out.append(item)
            continue
        out.append(_parse_scalar(payload))
        if index < len(lines) and lines[index].indent > indent:
            raise SimpleYamlError(f"scalar list item at line {line.number} cannot have a nested block")
    return out, index


def _split_key_value(content: str, line_no: int) -> Tuple[str, str, bool]:
    # 按第一个未被引号包住的冒号分割 key/value。
    in_single = False
    in_double = False
    for pos, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            key = content[:pos].strip()
            value = content[pos + 1 :].strip()
            if not key:
                raise SimpleYamlError(f"missing key before ':' at line {line_no}")
            return key, value, bool(value)
    raise SimpleYamlError(f"expected key: value at line {line_no}")


def _parse_scalar(raw: str) -> Any:
    # 先尝试解析成字面量类型，最后再退回普通字符串。
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
