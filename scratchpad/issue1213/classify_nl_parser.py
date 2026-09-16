#!/usr/bin/env python3
"""Line-level split of crates/discopt-core/src/nl_parser.rs:
format decoding vs ModelRepr construction (issue #1213, task 1).

Method
------
1. Scan top-level items with a brace-depth scanner -> (name, start, end).
2. Assign each item a role from an explicit table (every item must be listed;
   an unlisted item is a hard error, so the table cannot silently go stale).
3. For items marked FUSED, classify every *code* line individually:
   a line that names the IR (arena/ExprNode/ModelRepr/ConstraintRepr/VarInfo/
   ExprId/intern/...) is BUILD-coupled; a line that names only the lexer
   (reader/parse_f64/opcode/strip_prefix/...) is DECODE; anything else is GLUE.

Prints an executed-assertion count and exits non-zero if it is zero.
"""

import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[2] / "crates/discopt-core/src/nl_parser.rs"
text = SRC.read_text()
lines = text.splitlines()
N = len(lines)

# ---------------------------------------------------------------- item scan
# Depth scanner: an item opens at depth 0 and closes when depth returns to 0
# (or immediately, for a `;`-terminated declaration).
items = []
depth = 0
cur = None
attr_start = None
item_re = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?"
    r"(fn|struct|enum|impl|mod|const|static|type|trait)\b(.*)$"
)
for i, ln in enumerate(lines, start=1):
    stripped = ln.strip()
    if cur is None and depth == 0:
        if stripped.startswith("#["):
            if attr_start is None:
                attr_start = i
            continue
        if stripped == "" or stripped.startswith("//"):
            if stripped == "":
                attr_start = None
            continue
        m = item_re.match(stripped)
        if m:
            name = re.split(r"[\s<({:;]", m.group(2).strip())[0] or m.group(1)
            if m.group(1) == "impl":
                # `impl fmt::Display for NlParseError` -> key on the type implemented
                mm = re.search(r"for\s+([A-Za-z_][\w:]*)", stripped)
                if mm:
                    name = mm.group(1).split("::")[-1]
                else:
                    # inherent impl: `impl<'a> LineReader<'a> {`
                    mm = re.match(r"impl(?:<[^>]*>)?\s+([A-Za-z_][\w:]*)", stripped)
                    assert mm, f"cannot name impl at line {i}: {stripped}"
                    name = mm.group(1).split("::")[-1]
            else:
                name = re.sub(r"<.*", "", name)
            cur = {"kind": m.group(1), "name": name, "start": attr_start or i}
            attr_start = None
        else:
            attr_start = None
    opens, closes = ln.count("{"), ln.count("}")
    prev = depth
    depth += opens - closes
    if cur is not None:
        if depth == 0 and (opens or closes or stripped.endswith(";")):
            cur["end"] = i
            items.append(cur)
            cur = None
assert depth == 0, f"unbalanced braces, final depth {depth}"
assert cur is None, f"unterminated item {cur}"

# ---------------------------------------------------------------- roles
# DECODE  : .nl format lexing/structure only; knows nothing about the IR.
# BUILD   : ModelRepr / arena construction only; knows nothing about the file.
# FUSED   : does both in the same body (line-classified below).
# TESTS   : #[cfg(test)]
# SUPPORT : error type, public API wrappers, plain data carriers.
ROLES = {
    "NlParseError": "SUPPORT",
    "fmt": "SUPPORT",
    "std": "SUPPORT",
    "NlHeader": "DECODE",
    "LineReader": "DECODE",
    "parse_usize": "DECODE",
    "parse_i32": "DECODE",
    "parse_f64": "DECODE",
    "split_ws": "DECODE",
    "parse_header": "DECODE",
    "BinCursor": "DECODE",
    "fmt_f64": "DECODE",
    "opcode_arity": "DECODE",
    "transcode_expr": "DECODE",
    "bound_value_count": "DECODE",
    "transcode_binary_body": "DECODE",
    "transcode_binary_nl": "DECODE",
    "is_zero_constant": "BUILD",
    "parse_expr": "FUSED",
    "parse_opcode": "FUSED",
    "parse_nl_full": "FUSED",
    "ParsedNl": "SUPPORT",
    "parse_nl": "SUPPORT",
    "parse_nl_with_complementarity": "SUPPORT",
    "parse_nl_file": "SUPPORT",
    "parse_nl_file_with_complementarity": "SUPPORT",
    "parse_nl_file_full": "SUPPORT",
    "tests": "TESTS",
}

checks = 0
unknown = [it for it in items if it["name"] not in ROLES]
if unknown:
    sys.exit(
        "unclassified top-level items (update ROLES): "
        + ", ".join(f"{i['name']}@{i['start']}" for i in unknown)
    )

# ---------------------------------------------------------------- line class
# A line is IR-COUPLED if it names a discopt IR type or the arena. Such a line
# cannot move into a format-only decoder crate as written. Everything else in a
# parsing function is format handling or control flow.
IR = re.compile(
    r"\b(arena|ExprArena|ExprNode|ExprId|ModelRepr|ConstraintRepr|"
    r"ConstraintSense|ObjectiveSense|VarInfo|VarType|BinOp|UnOp|"
    r"MathFunc|ComplementarityRepr|ParsedNl|var_nodes|intern)\b"
)


def codeline(s):
    t = s.strip()
    return bool(t) and not t.startswith("//")


tally = {
    "DECODE": 0,
    "BUILD": 0,
    "FUSED_ir": 0,
    "FUSED_free": 0,
    "SUPPORT": 0,
    "TESTS": 0,
    "OUTSIDE": 0,
}
covered = [False] * (N + 1)
per_item, transitions, leaks = [], 0, []

for it in items:
    role = ROLES[it["name"]]
    body = range(it["start"], it["end"] + 1)
    for ln in body:
        covered[ln] = True
    code = [ln for ln in body if codeline(lines[ln - 1])]
    checks += 1
    if role == "DECODE":
        # Contract: a DECODE item must be IR-free, otherwise it is not movable.
        bad = [ln for ln in code if IR.search(lines[ln - 1])]
        checks += len(code)
        if bad:
            leaks.append((it["name"], bad))
        tally["DECODE"] += len(code)
        per_item.append(
            (
                it["name"],
                it["start"],
                it["end"],
                len(code),
                "DECODE (IR-free)" if not bad else f"DECODE +{len(bad)} IR LEAK",
            )
        )
    elif role == "FUSED":
        ir = free = 0
        prev = None
        for ln in code:
            hit = bool(IR.search(lines[ln - 1]))
            checks += 1
            if hit:
                ir += 1
            else:
                free += 1
            if prev is not None and hit != prev:
                transitions += 1
            prev = hit
        tally["FUSED_ir"] += ir
        tally["FUSED_free"] += free
        per_item.append(
            (it["name"], it["start"], it["end"], len(code), f"FUSED  ir={ir} ir-free={free}")
        )
    else:
        tally[role] += len(code)
        per_item.append((it["name"], it["start"], it["end"], len(code), role))

for ln in range(1, N + 1):
    if not covered[ln] and codeline(lines[ln - 1]):
        tally["OUTSIDE"] += 1

print(f"file: {SRC}")
print(f"total lines: {N}")
print()
print(f"{'item':<38}{'span':>10}{'code':>7}  role")
for name, s_, e_, c, role in per_item:
    print(f"{name:<38}{str(s_) + '-' + str(e_):>10}{c:>7}  {role}")
print()
print("--- code-line tally (comments/blank excluded) ---")
for k, v in tally.items():
    print(f"  {k:<14}{v:>6}")
print(f"  {'TOTAL':<14}{sum(tally.values()):>6}")
non_test = sum(v for k, v in tally.items() if k != "TESTS")
movable = tally["DECODE"]
fused = tally["FUSED_ir"] + tally["FUSED_free"]
print()
print(f"non-test code lines                      : {non_test}")
print(f"IR-free decoding, movable as-is          : {movable}  ({100 * movable / non_test:.1f}%)")
TRANSCODER = {
    "BinCursor",
    "fmt_f64",
    "opcode_arity",
    "transcode_expr",
    "bound_value_count",
    "transcode_binary_body",
    "transcode_binary_nl",
}
transcoder_lines = sum(c for n, _, _, c, _ in per_item if n in TRANSCODER)
print(f"  of which binary->text transcoder       : {transcoder_lines}")
print(f"fused decode+build functions             : {fused}  ({100 * fused / non_test:.1f}%)")
print(f"  IR-coupled lines inside them           : {tally['FUSED_ir']}")
print(f"  IR-free lines inside them              : {tally['FUSED_free']}")
print(f"  coupled<->free transitions (interleave): {transitions}")
print(f"public API wrappers / error type / data  : {tally['SUPPORT']}")
print(f"test module                              : {tally['TESTS']}")
print()
if leaks:
    print("IR LEAKS in supposedly format-only items:")
    for n, b in leaks:
        print(f"  {n}: lines {b[:10]}")
print(f"executed classifications: {checks}")
if checks == 0:
    sys.exit("PROBE DID NOT FIRE: zero classifications")
if leaks:
    sys.exit("DECODE items are not IR-free; the movable-surface number is invalid")
