"""Search text for patterns with the Aho-Corasick algorithm."""

from __future__ import annotations

import argparse
from collections import deque
from functools import wraps
from pathlib import Path
import sys
import time
from typing import Callable, Iterable, NamedTuple, ParamSpec, Sequence, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class SearchMatch(NamedTuple):
    """A match span in the original text, with a zero-based half-open range."""

    start: int
    end: int
    pattern: str


def _log_execution_time(function: Callable[P, R]) -> Callable[P, R]:
    """Print a function's execution time to stderr."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            print(
                f"[time] {function.__name__}: {time.perf_counter() - started:.6f}s",
                file=sys.stderr,
            )

    return wrapper


def _patterns_from(value: str | Iterable[str]) -> list[str]:
    """Validate and deduplicate a pattern string or iterable."""
    if isinstance(value, str):
        patterns = [value]
    else:
        try:
            patterns = list(value)
        except TypeError as error:
            raise TypeError("sub_string must be a string or an iterable of strings") from error
    if not patterns:
        raise ValueError("at least one non-empty pattern is required")
    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError("every pattern must be a string")
    if any(not pattern for pattern in patterns):
        raise ValueError("empty patterns are not supported")
    return list(dict.fromkeys(patterns))


def _validate_options(
    text: str,
    case_sensitivity: bool,
    method: str,
    count: int | None,
) -> None:
    if not isinstance(text, str):
        raise TypeError("string must be a string")
    if not isinstance(case_sensitivity, bool):
        raise TypeError("case_sensitivity must be a boolean")
    if method not in {"first", "last"}:
        raise ValueError("method must be 'first' or 'last'")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer or None")
        if count < 0:
            raise ValueError("count cannot be negative")


def _find_matches(
    text: str,
    patterns: Sequence[str],
    case_sensitivity: bool,
) -> list[SearchMatch]:
    """Build the trie and failure links, then return every overlapping match."""
    normalized_patterns = [
        pattern if case_sensitivity else pattern.casefold()
        for pattern in patterns
    ]
    if case_sensitivity:
        normalized_text = text
        positions = list(range(len(text)))
    else:
        folded = [(char.casefold(), index) for index, char in enumerate(text)]
        normalized_text = "".join(part for part, _ in folded)
        positions = [index for part, index in folded for _ in part]

    # Each state stores character transitions, a failure state, and pattern IDs.
    transitions: list[dict[str, int]] = [{}]
    failures = [0]
    outputs: list[list[int]] = [[]]
    for pattern_id, pattern in enumerate(normalized_patterns):
        state = 0
        for char in pattern:
            if char not in transitions[state]:
                transitions[state][char] = len(transitions)
                transitions.append({})
                failures.append(0)
                outputs.append([])
            state = transitions[state][char]
        outputs[state].append(pattern_id)

    queue = deque(transitions[0].values())
    while queue:
        state = queue.popleft()
        for char, child in transitions[state].items():
            queue.append(child)
            fallback = failures[state]
            while fallback and char not in transitions[fallback]:
                fallback = failures[fallback]
            failures[child] = transitions[fallback].get(char, 0)
            outputs[child].extend(outputs[failures[child]])

    matches: list[SearchMatch] = []
    state = 0
    for index, char in enumerate(normalized_text):
        while state and char not in transitions[state]:
            state = failures[state]
        state = transitions[state].get(char, 0)
        for pattern_id in outputs[state]:
            start = index - len(normalized_patterns[pattern_id]) + 1
            if start >= 0:
                matches.append(SearchMatch(
                    positions[start],
                    positions[index] + 1,
                    patterns[pattern_id],
                ))
    return sorted(matches, key=lambda match: (match.start, match.end, match.pattern))


@_log_execution_time
def search(
    string: str,
    sub_string: str | Iterable[str],
    case_sensitivity: bool = False,
    method: str = "first",
    count: int | None = None,
) -> tuple[int, ...] | dict[str, tuple[int, ...] | None] | None:
    """Return match start indexes, optionally grouped by pattern.

    ``count`` limits matches globally. ``last`` selects the final matches and
    returns each pattern's indexes in reverse order.
    """
    _validate_options(string, case_sensitivity, method, count)
    single_pattern = isinstance(sub_string, str)
    patterns = _patterns_from(sub_string)
    matches = _find_matches(string, patterns, case_sensitivity)
    if not matches:
        return None
    if count == 0:
        return () if single_pattern else {pattern: None for pattern in patterns}
    if count is not None:
        matches = matches[:count] if method == "first" else matches[-count:]

    indexes: dict[str, list[int]] = {pattern: [] for pattern in patterns}
    for match in matches:
        indexes[match.pattern].append(match.start)
    result = {
        pattern: tuple(reversed(found)) if method == "last" else tuple(found)
        for pattern, found in indexes.items()
    }
    if single_pattern:
        return result[patterns[0]] or None
    return {pattern: found or None for pattern, found in result.items()}


_COLORS = ("\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m")
_RESET = "\033[0m"


def _highlight(text: str, matches: Sequence[SearchMatch], color: bool) -> str:
    """Color match spans by pattern, preserving overlapping highlights."""
    if not color or not matches:
        return text
    events: dict[int, list[tuple[bool, int]]] = {}
    colors: dict[str, int] = {}
    for match in matches:
        color_id = colors.setdefault(match.pattern, len(colors) % len(_COLORS))
        events.setdefault(match.start, []).append((True, color_id))
        events.setdefault(match.end, []).append((False, color_id))

    result: list[str] = []
    active: list[int] = []
    for index, char in enumerate(text):
        for is_start, color_id in events.get(index, []):
            if is_start:
                active.append(color_id)
            elif color_id in active:
                active.remove(color_id)
        result.append(f"{_COLORS[active[-1]]}{char}{_RESET}" if active else char)
    return "".join(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find occurrences of one or more patterns with Aho-Corasick."
    )
    parser.add_argument("text", nargs="?", help="source text")
    parser.add_argument("patterns", nargs="*", help="patterns to search for")
    parser.add_argument("-p", "--pattern", action="append", default=[], dest="extra_patterns")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-f", "--file", type=Path, help="read UTF-8 source text from a file")
    source.add_argument("-t", "--text", dest="option_text", help="source text")
    parser.add_argument("-s", "--case-sensitive", action="store_true")
    parser.add_argument("-m", "--method", choices=("first", "last"), default="first")
    parser.add_argument("-k", "--count", type=int, help="maximum number of matches")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI highlighting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line search and print up to ten output lines."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.file is not None:
            text = args.file.read_text(encoding="utf-8")
        elif args.option_text is not None:
            text = args.option_text
        else:
            text = args.text
        if text is None:
            parser.error("provide source text or use --file")

        positional_patterns = [args.text] if (
            args.file is not None or args.option_text is not None
        ) and args.text is not None else []
        patterns = [*positional_patterns, *args.patterns, *args.extra_patterns]
        if not patterns:
            parser.error("provide at least one pattern")
        _validate_options(text, args.case_sensitive, args.method, args.count)
        patterns = _patterns_from(patterns)
        matches = _find_matches(text, patterns, args.case_sensitive)
        if args.count is not None:
            matches = matches[:args.count] if args.method == "first" else (
                matches[-args.count:] if args.count else []
            )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Найдено совпадений: {len(matches)}")
    if not matches:
        print("Совпадений не найдено.")
        return 0
    highlighted = _highlight(text, matches, not args.no_color and sys.stdout.isatty())
    escaped_text = highlighted.replace("\r", "\\r").replace("\n", "\\n")
    print(f"Текст: {escaped_text}")
    visible = matches[:7 if len(matches) > 8 else 8]
    for match in visible:
        print(f"[{match.start}:{match.end}] {match.pattern!r}")
    if len(matches) > len(visible):
        print(f"... вывод ограничен 10 строками; скрыто совпадений: {len(matches) - len(visible)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
