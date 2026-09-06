import ast, os
from pathlib import Path
PKG = Path("python/discopt")
ROLE1 = {"time_limit", "_outer_budget", "_remaining_tl", "_node_remaining",
         "remaining", "_root_remaining", "_probe_remaining", "_ms_remaining",
         "_remaining", "_hg_remaining", "time_budget", "total_time_limit"}
SAT_CALLS = {"saturate_role2", "_role2_saturate"}
#: wrappers that return their argument unchanged (or a no-clock value)
PASSTHRU = {"_role2_horizon", "_role2_budget", "_role2_deadline"}

def bounded(n):
    if isinstance(n, ast.Constant):
        return isinstance(n.value, (int, float))
    if isinstance(n, ast.Name):
        return n.id.lstrip("_").isupper()
    if isinstance(n, ast.Call):
        f = n.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        if name in SAT_CALLS:
            return True
        if name in PASSTHRU:
            return bool(n.args) and bounded(n.args[0])
        if name == "min":
            return any(bounded(a) for a in n.args)
        if name == "max":
            return bool(n.args) and all(bounded(a) for a in n.args)
        if name in ("float", "abs"):
            return all(bounded(a) for a in n.args)
        return False
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div, ast.Add, ast.Sub)):
        return bounded(n.left) and bounded(n.right)
    if isinstance(n, ast.IfExp):
        return bounded(n.body) and bounded(n.orelse)
    return False

rows=[]
for root, dirs, files in os.walk(PKG):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in sorted(files):
        if not f.endswith(".py"): continue
        p = Path(root)/f
        src = p.read_text().splitlines()
        tree = ast.parse("\n".join(src), filename=str(p))
        parents = {}
        for n in ast.walk(tree):
            for c in ast.iter_child_nodes(n):
                parents[c] = n
        seen=set()
        for n in ast.walk(tree):
            if not (isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div))):
                continue
            names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
            if not (names & ROLE1):
                continue
            # widen to the largest enclosing expression (stop at the statement)
            top = n
            while (top in parents and isinstance(parents[top], ast.expr)
                   and not isinstance(parents[top], (ast.Compare, ast.BoolOp))):
                top = parents[top]
            key = (str(p), top.lineno)
            if key in seen:
                continue
            seen.add(key)
            rows.append((str(p.relative_to(PKG)), top.lineno,
                         src[top.lineno-1].strip(), bounded(top)))
for r in sorted(rows):
    print(f"{'SAT ' if r[3] else 'GROW'} {r[0]}:{r[1]}  {r[2]}")
print(f"# sites={len(rows)} grow={sum(1 for r in rows if not r[3])}")
