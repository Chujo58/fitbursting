from pathlib import Path
import subprocess
import sys
import re
from dataclasses import dataclass
from typing import Callable
import textwrap

from colorama import (
    Style as _Style,
    Fore as _Fore,
    Back as _Back,
    just_fix_windows_console,
)

just_fix_windows_console()

Style = _Style
Fore = _Fore
Back = _Back

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class Color:
    @staticmethod
    def fg(n: int):
        return f"\033[38;5;{n}m"

    @staticmethod
    def bg(n: int):
        return f"\033[48;5;{n}m"

    @staticmethod
    def rgb(r: int, g: int, b: int):
        return f"\033[38;2;{r};{g};{b}m"


def clear():
    subprocess.run("clear")


def reset():
    print(Style.RESET_ALL, end="")


def contentWidth(content):
    content_width = 0
    if isinstance(content, str):
        content_width = visible_width(content)

    if isinstance(content, list):
        for line in content:
            content_width = max(visible_width(line), content_width)

    return content_width


def visible_width(text: str) -> int:
    return len(ANSI_ESCAPE_RE.sub("", text))


def box(title: str, content: str | list, padding: int, min_width: int, color=None):
    print("\n".join(box_lines(title, content, padding, min_width, color)))


def box_lines(
    title: str, content: str | list, padding: int, min_width: int, color=None
):
    content_width = contentWidth(content)

    inner_width = max(content_width, min_width) + (padding * 2)
    if len(title) > 0:
        title = f" {title} "

    if color is None:
        color = ""

    topLine = (
        color
        + "╭─"
        + Style.RESET_ALL
        + title
        + color
        + "─" * (inner_width - len(title) - 1)
        + "╮"
        + Style.RESET_ALL
    )
    botLine = color + "╰" + "─" * inner_width + "╯"

    lines = [topLine]
    if isinstance(content, str):
        lines.append(_format_box_content(content, padding, inner_width, color))
    else:
        for line in content:
            lines.append(_format_box_content(line, padding, inner_width, color))

    lines.append(botLine)
    return lines


def _print_box_content(content, padding, inner_width, color):
    print(_format_box_content(content, padding, inner_width, color))


def _format_box_content(content, padding, inner_width, color):
    content_width = visible_width(content)
    return (
        color
        + "│"
        + Style.RESET_ALL
        + " " * padding
        + content
        + " " * (inner_width - padding - content_width)
        + color
        + "│"
        + Style.RESET_ALL
    )


def seperator(width):
    print("─" * width)


@dataclass
class SelectionOption:
    name: str
    keybind: str
    callback: Callable[[], None]


def selection_box(
    title: str,
    options: list[SelectionOption],
    padding: int = 2,
    min_width: int = 0,
    color=None,
):
    option_lines = [
        f"{Color.fg(2) + option.keybind + Style.RESET_ALL} ⋈┈◎ {option.name}"
        for option in options
    ]
    box(title, option_lines, padding, min_width, color)


def run_selection_option(keybind: str, options: list[SelectionOption]) -> bool:
    for option in options:
        if option.keybind == keybind:
            option.callback()
            return True

    return False


def ask_user_input(options: list[SelectionOption]):
    key = input("> ").strip().lower()
    run_selection_option(key, options)


def progress_bar(
    current: int,
    total: int,
    width: int = 40,
    label: str = "",
    color=None,
    show_percent: bool = True,
    end: str = "\r",
):
    if total <= 0:
        total = 1

    if color is None:
        color = ""

    progress = max(0.0, min(current / total, 1.0))
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percent = f" ({progress * 100:.1f}%)" if show_percent else ""
    prefix = f"{label} " if label else ""

    sys.stdout.write("\r\033[K")

    sys.stdout.write(
        f"{prefix}{color}{bar}{Style.RESET_ALL} {current}/{total}{percent}{end}"
    )
    sys.stdout.flush()


def table(
    headers: list[str],
    rows: list[list[object]],
    padding: int = 2,
    min_column_width: int = 0,
    color=None,
    header_color=None,
    show_separator: bool = True,
):
    print(
        "\n".join(
            table_lines(
                headers,
                rows,
                padding=padding,
                min_column_width=min_column_width,
                color=color,
                header_color=header_color,
                show_separator=show_separator,
            )
        )
    )


def table_lines(
    headers: list[str],
    rows: list[list[object]],
    padding: int = 2,
    min_column_width: int = 0,
    color=None,  # now: str OR list[str] (per-column)
    header_color=None,  # same: str OR list[str]
    show_separator: bool = True,
    sep_color=None,
    col_size: list[int] = None,
):
    normalized_headers = [str(h) for h in headers]
    normalized_rows = [[str(c) for c in row] for row in rows]

    column_count = max(
        [len(normalized_headers), *[len(row) for row in normalized_rows]], default=0
    )

    body_colors = _normalize_colors(color, column_count)
    header_colors = _normalize_colors(
        header_color if header_color is not None else color, column_count
    )

    # --- width calculation, now col_size-aware ---
    widths: list[int] = []
    for column_index in range(column_count):
        fixed = None
        if col_size is not None and column_index < len(col_size):
            fixed = col_size[column_index]

        if fixed:  # truthy, non-zero, non-None
            widths.append(max(fixed, min_column_width))
            continue

        column_cells = [
            row[column_index] for row in normalized_rows if column_index < len(row)
        ]
        header_width = (
            visible_width(normalized_headers[column_index])
            if column_index < len(normalized_headers)
            else 0
        )
        cell_width = max(
            [header_width, *(visible_width(cell) for cell in column_cells)], default=0
        )
        widths.append(max(cell_width, min_column_width))

    def wrap_cell(text: str, width: int) -> list[str]:
        return ansi_aware_wrap(text, width)

    def format_multiline_row(cells: list[str], cell_colors: list[str]) -> list[str]:
        wrapped_columns = []
        for column_index, width in enumerate(widths):
            cell = cells[column_index] if column_index < len(cells) else ""
            wrapped_columns.append(wrap_cell(cell, width))

        height = max((len(col) for col in wrapped_columns), default=0)
        out_lines = []
        for line_index in range(height):
            padded_cells = []
            for column_index, width in enumerate(widths):
                col_lines = wrapped_columns[column_index]
                cell_line = col_lines[line_index] if line_index < len(col_lines) else ""
                padded = cell_line + (" " * max(width - visible_width(cell_line), 0))
                col_color = (
                    cell_colors[column_index] if column_index < len(cell_colors) else ""
                )
                if col_color:
                    padded = col_color + padded + Style.RESET_ALL
                padded_cells.append(padded)
            out_lines.append((" " * padding).join(padded_cells))
        return out_lines

    lines: list[str] = []
    if normalized_headers:
        lines.extend(format_multiline_row(normalized_headers, header_colors))
    if show_separator and widths:
        separator_line = (" " * padding).join("─" * w for w in widths)
        lines.append(
            sep_color + separator_line + Style.RESET_ALL
            if sep_color
            else separator_line
        )
    for row in normalized_rows:
        lines.extend(format_multiline_row(row, body_colors))

    return lines


def _normalize_colors(color, column_count: int) -> list[str]:
    """Accepts a single color string (applies to every column) or a
    list of per-column colors. Always returns a list of length
    column_count."""
    if color is None:
        color = ""
    if isinstance(color, str):
        return [color] * column_count
    # it's a list/sequence — pad with "" for missing trailing columns
    return [color[i] if i < len(color) else "" for i in range(column_count)]


def ansi_aware_wrap(text: str, width: int) -> list[str]:
    """Wrap text to `width` visible columns, never splitting an ANSI
    escape sequence and never counting it toward the width."""
    if width <= 0:
        return [text]

    # Fast path: no escape codes, no need for manual walking.
    if not ANSI_ESCAPE_RE.search(text):
        return textwrap.wrap(text, width=width) or [""]

    # Tokenize into escape-codes and visible "words" (so we still get
    # word-boundary wrapping rather than raw character chopping).
    tokens = []
    last_end = 0
    for match in re.finditer(r"(\x1b\[[0-9;]*m)|(\S+)|(\s+)", text):
        tokens.append(match.group(0))

    lines: list[str] = []
    current_line = ""
    current_visible = 0
    pending_codes = ""  # escape codes waiting to attach to next visible chunk

    def visible_len(token: str) -> int:
        return 0 if ANSI_ESCAPE_RE.fullmatch(token) else len(token)

    for token in tokens:
        if ANSI_ESCAPE_RE.fullmatch(token):
            # Escape codes never break a line and never cost width.
            current_line += token
            pending_codes += token
            continue

        token_visible = visible_len(token)
        is_space = token.isspace()

        if (
            current_visible + token_visible > width
            and current_visible > 0
            and not is_space
        ):
            # Wrap: push current line, start a new one.
            lines.append(current_line.rstrip())
            current_line = pending_codes + token  # carry active color forward
            current_visible = token_visible
        else:
            current_line += token
            current_visible += token_visible

        pending_codes = ""

    if current_line.strip() or not lines:
        lines.append(current_line.rstrip())

    return lines or [""]


def side_by_side(left_lines: list[str], right_lines: list[str], gap: int = 4):
    left_width = max((visible_width(line) for line in left_lines), default=0)
    right_width = max((visible_width(line) for line in right_lines), default=0)
    total_lines = max(len(left_lines), len(right_lines))

    for index in range(total_lines):
        left_line = left_lines[index] if index < len(left_lines) else ""
        right_line = right_lines[index] if index < len(right_lines) else ""
        print(
            left_line
            + (" " * max(left_width - visible_width(left_line), 0))
            + (" " * gap)
            + right_line
            + (" " * max(right_width - visible_width(right_line), 0))
        )


def pick_folder(start_path: str = str(Path.home())) -> str | None:
    state = {
        "current": Path(start_path).expanduser().resolve(),
        "selected": None,
        "exit": False,
    }

    def refresh_dirs():
        current = state["current"]
        dirs = sorted(
            [p for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")]
        )
        return dirs

    while True:
        current = state["current"]
        dirs = refresh_dirs()

        options: list[SelectionOption] = []

        # ── shortcuts ─────────────────────────────
        shortcuts = {
            "h": Path.home(),
            "d": Path.home() / "Downloads",
        }

        for key, path in shortcuts.items():
            if path.exists():
                options.append(
                    SelectionOption(
                        name=f"{path.name if path.name else path} ({path})",
                        keybind=key,
                        callback=lambda p=path: state.update(current=p),
                    )
                )

        # ── navigation ────────────────────────────
        if current.parent != current:
            options.append(
                SelectionOption(
                    name=".. (Go Up)",
                    keybind="u",
                    callback=lambda: state.update(current=current.parent),
                )
            )

        # ── directories ───────────────────────────
        for i, d in enumerate(dirs[:9]):
            options.append(
                SelectionOption(
                    name=d.name,
                    keybind=str(i),
                    callback=lambda p=d: state.update(current=p),
                )
            )

        # ── actions ───────────────────────────────
        options.append(
            SelectionOption(
                name=f"Select this folder → {current}",
                keybind="s",
                callback=lambda: state.update(selected=str(current)),
            )
        )

        options.append(
            SelectionOption(
                name="Cancel", keybind="q", callback=lambda: state.update(exit=True)
            )
        )

        # ── render ────────────────────────────────
        selection_box(f"Pick install directory: {current}", options)

        key = input("\n> ").strip()

        run_selection_option(key, options)

        if state["exit"]:
            return None

        if state["selected"]:
            return state["selected"]
