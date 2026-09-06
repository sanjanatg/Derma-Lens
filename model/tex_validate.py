"""
tex_validate.py — structural LaTeX validator (no toolchain required)
=====================================================================
When no LaTeX distribution is installed, this catches the class of errors a
text grep misses and a compiler would report:

  1. \\begin{env}/\\end{env} balance (with nesting stack and line numbers)
  2. every \\ref{...} / \\eqref resolves to a \\label; every label is unique
  3. every \\cite{...} resolves to a \\bibitem; every \\bibitem is cited
  4. unbalanced braces outside comments ({} and $..$ math toggles)

Usage:  python tex_validate.py [path/to/paper.tex]
Exit code 0 = clean, 1 = errors found.
"""

import re
import sys
from pathlib import Path

DEFAULT_TEX = Path(__file__).resolve().parent.parent / "ppt_files" / \
    "dermalens_paper.tex"

CMD_ARG = re.compile(
    r"\\(?:ref|eqref|label|cite|bibitem)\s*(?:\[[^\]]*\])?\{([^}]*)\}")
BEGIN_RE = re.compile(r"\\begin\{(\w+\*?)\}")
END_RE = re.compile(r"\\end\{(\w+\*?)\}")


def strip_comments(text):
    """Remove % comments (respecting escaped \\%)."""
    out = []
    for line in text.splitlines():
        res, i = [], 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                res.append(line[i:i + 2])
                i += 2
                continue
            if line[i] == "%":
                break
            res.append(line[i])
            i += 1
        out.append("".join(res))
    return "\n".join(out)


def main():
    tex_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEX
    raw = tex_path.read_text(encoding="utf-8")
    text = strip_comments(raw)
    errors, warnings = [], []

    # 1. environment balance with a nesting stack
    tokens = [(m.start(), m.group(1), "begin") for m in BEGIN_RE.finditer(text)]
    tokens += [(m.start(), m.group(1), "end") for m in END_RE.finditer(text)]
    tokens.sort()
    stack = []
    for pos, env, kind in tokens:
        line = text.count("\n", 0, pos) + 1
        if kind == "begin":
            stack.append((env, line))
        else:
            if not stack:
                errors.append(f"line {line}: \\end{{{env}}} with empty stack")
            elif stack[-1][0] != env:
                errors.append(f"line {line}: \\end{{{env}}} but stack top is "
                              f"\\begin{{{stack[-1][0]}}} (line {stack[-1][1]})")
                # recover: pop until match
                names = [e for e, _ in stack]
                if env in names:
                    while stack and stack[-1][0] != env:
                        stack.pop()
                    stack.pop()
            else:
                stack.pop()
    for env, line in stack:
        errors.append(f"line {line}: \\begin{{{env}}} never closed")

    # 2. labels vs refs
    labels = CMD_ARG.findall(text)
    lab_map, seen = {}, set()
    # classify by which command captured them: re-scan with positions
    for m in CMD_ARG.finditer(text):
        kind = text[m.start():m.start() + 8]
        key, val = m.group(1).strip(), m.group(1).strip()
        if "\\label" in text[max(0, m.start() - 8):m.start() + 8]:
            if key in seen:
                errors.append(f"duplicate label: {key}")
            seen.add(key)
            lab_map[key] = text.count("\n", 0, m.start()) + 1

    refs, cites, bibitems = set(), set(), set()
    for m in re.finditer(r"\\(?:ref|eqref)\s*\{([^}]*)\}", text):
        refs.add(m.group(1).strip())
    for m in re.finditer(r"\\cite\s*(?:\[[^\]]*\])?\{([^}]*)\}", text):
        for c in m.group(1).split(","):
            cites.add(c.strip())
    for m in re.finditer(r"\\label\s*\{([^}]*)\}", text):
        pass  # already in lab_map
    for m in re.finditer(r"\\bibitem\s*(?:\[[^\]]*\])?\{([^}]*)\}", text):
        bibitems.add(m.group(1).strip())

    for r in sorted(refs):
        if r not in lab_map:
            errors.append(f"\\ref{{{r}}} has no matching \\label")
    for c in sorted(cites):
        if c not in bibitems:
            errors.append(f"\\cite{{{c}}} has no matching \\bibitem")
    for b in sorted(bibitems - cites):
        warnings.append(f"\\bibitem{{{b}}} never cited")

    # 3. brace balance (outside verbatim-ish envs, comments already stripped)
    depth, line = 0, 1
    for ch in text:
        if ch == "\n":
            line += 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"line {line}: unbalanced closing brace")
                depth = 0
    if depth != 0:
        errors.append(f"unbalanced braces: net depth {depth} at EOF")

    # 4. $ math toggle parity per paragraph (rough check, skip $$)
    body = text.replace("$$", "")
    if body.count("$") % 2:
        errors.append("odd number of $ math toggles")

    print(f"validating {tex_path}")
    print(f"  environments: {sum(1 for _,_,k in tokens if k=='begin')} begin / "
          f"{sum(1 for _,_,k in tokens if k=='end')} end")
    print(f"  labels: {len(lab_map)}, refs: {len(refs)}, "
          f"cites: {len(cites)}, bibitems: {len(bibitems)}")
    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if not errors:
        print("STRUCTURAL VALIDATION PASSED")
        return 0
    print("STRUCTURAL VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
