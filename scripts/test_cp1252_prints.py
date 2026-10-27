#!/usr/bin/env python3
"""Every print / logging string literal in the product must encode as cp1252.

The Windows console (and a child process whose stdout is a pipe) uses the
ANSI code page, cp1252 on the dev PC. A print of a character outside it
raises ``UnicodeEncodeError`` and crashes the caller; the ``→`` in the
beamngpy attach print did exactly that. This scan is static (``ast``): it
looks at the string literals passed to ``print(...)`` and to logging calls
(``logging.x(...)`` / ``<logger>.x(...)``), and the ``help=`` keyword of any
call (argparse ``add_argument``: ``--help`` prints it), in every
``python/**/*.py`` file and the ``python/run_vision.py`` entry. An arrow (``→``) is reported by
name; anything else outside cp1252 is reported with its code point.

The guard is tested on in-memory samples first, so a scan that matches
nothing cannot pass silently.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "python" / "run_vision.py"
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
LOGGER_NAME = re.compile(r"(?i)^_*(log|logger|logging)$|(?i:_log|_logger)$")
ARROW = "\u2192"


def _is_output_call(node: ast.Call) -> bool:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id == "print"
    if isinstance(fn, ast.Attribute) and fn.attr in LOG_METHODS:
        base = fn.value
        if isinstance(base, ast.Name):
            return bool(LOGGER_NAME.search(base.id))
        if isinstance(base, ast.Attribute):
            return bool(LOGGER_NAME.search(base.attr))
        if isinstance(base, ast.Call):  # logging.getLogger(...).warning(...)
            inner = base.func
            return isinstance(inner, ast.Attribute) and inner.attr == "getLogger"
    return False


def _literals(call: ast.Call):
    """String literals this call can print: every argument of print / logging, else ``help=`` only."""
    args = [*call.args, *(kw.value for kw in call.keywords)] if _is_output_call(call) else [
        kw.value for kw in call.keywords if kw.arg == "help"]
    for arg in args:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield sub.lineno, sub.value


def scan_source(text: str, label: str) -> list[str]:
    """Problems in one module's print / logging literals, as ``label:line: reason`` strings."""
    problems: list[str] = []
    tree = ast.parse(text, filename=label)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for lineno, value in _literals(node):
            if ARROW in value:
                problems.append(f"{label}:{lineno}: '\u2192' (U+2192 arrow) in {value!r}; use '->'")
                continue
            try:
                value.encode("cp1252")
            except UnicodeEncodeError as exc:
                bad = value[exc.start]
                problems.append(f"{label}:{lineno}: U+{ord(bad):04X} does not encode as cp1252 in {value!r}")
    return problems


def product_files() -> list[Path]:
    files = sorted(p for p in (ROOT / "python").rglob("*.py") if "__pycache__" not in p.parts)
    if ENTRY not in files:
        files.append(ENTRY)
    return files


def check_guard_flags_samples() -> None:
    arrow = scan_source('print("GVD\u2192BeamNG attach")\n', "sample_arrow.py")
    assert len(arrow) == 1 and "U+2192 arrow" in arrow[0], arrow
    fstr = scan_source('x = 1\nprint(f"cam {x} \u2192 ok")\n', "sample_fstring.py")
    assert len(fstr) == 1 and "U+2192 arrow" in fstr[0], fstr
    logged = scan_source('import logging\n_LOG = logging.getLogger("t")\n_LOG.warning("a \u2192 b")\n', "sample_log.py")
    assert len(logged) == 1 and "U+2192 arrow" in logged[0], logged
    helped = scan_source('import argparse\nap = argparse.ArgumentParser()\n'
                         'ap.add_argument("--viz", help="auto\u2192non-primary")\n', "sample_help.py")
    assert len(helped) == 1 and "U+2192 arrow" in helped[0], helped
    helped_other = scan_source('ap.add_argument("--x", help=f"{1} done \u2713")\n', "sample_help_fstring.py")
    assert len(helped_other) == 1 and "U+2713" in helped_other[0], helped_other
    not_help = scan_source('ap.add_argument("--y", default="a\u2192b", metavar="\u2192")\n', "sample_not_help.py")
    assert not_help == [], not_help
    other = scan_source('print("done \u2713")\n', "sample_check.py")
    assert len(other) == 1 and "U+2713" in other[0], other
    ok = scan_source('print("retail \u2014 fullscreen", sep=" ")\nlabel = "not printed \u2192"\n', "sample_ok.py")
    assert ok == [], ok
    print("guard: arrow / f-string / logging / argparse help= / non-cp1252 samples flagged; "
          "em dash, non-help keywords and non-output strings pass")


def check_product_literals() -> None:
    files = product_files()
    assert ENTRY in files and ENTRY.is_file(), ENTRY
    problems: list[str] = []
    for path in files:
        problems += scan_source(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix())
    if problems:
        raise AssertionError("print / logging / help= literals that do not encode as cp1252:\n" + "\n".join(problems))
    print(f"product: {len(files)} files scanned, every print / logging / help= literal encodes as cp1252")


def main() -> None:
    check_guard_flags_samples()
    check_product_literals()
    print("test_cp1252_prints: OK")


if __name__ == "__main__":
    main()
