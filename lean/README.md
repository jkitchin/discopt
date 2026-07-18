# discopt certificate checker (Lean 4)

A Lean-verified checker for discopt **solution certificates**. Phase 0 ships the
**Tier-1 feasibility** checker: given a certificate emitted by
`discopt.certificate`, it proves — in the Lean kernel — that the reported
incumbent is genuinely feasible and attains the reported objective value.

See `docs/dev/lean-certificate-plan.md` for the full design (tiers, schema, the
roadmap to global-optimality certificates).

## What is proved

`Discopt/Model.lean` defines the model semantics over exact rationals (`Rat`) and
the theorem

```lean
theorem checkFeasible_sound (cert : Certificate) :
    checkFeasible cert = true → Feasible cert ∧ ObjOk cert
```

`Feasible`/`ObjOk` are *semantic* propositions (real `≤` over `Rat`,
`eval = some v`); `checkFeasible` is `decide (Feasible ∧ ObjOk)`, so soundness is
`of_decide_eq_true` — there is no gap between what the procedure checks and what
the theorem proves. Per certificate the kernel only *evaluates* the decision
procedure; it never searches for a proof, so checking scales to large models.

Only the Lean 4 **core** toolchain is used — **no Mathlib** — so the trusted base
is just the kernel. Transcendental functions (`exp`, `log`, …) are not evaluable
in exact `Rat`; a certificate whose checked expressions contain them is rejected
(that is Phase 1, which adds interval enclosures over Mathlib reals).

## Build & run

```bash
cd lean
lake build                       # compiles the library + `check` executable
lake exe check path/to/cert.json # -> "FEASIBLE" (exit 0) or "REJECTED" (exit 1)
```

Confirm no proof is stubbed:

```bash
lake env lean --run - <<'EOF'
import Discopt.Model
#print axioms Discopt.checkFeasible_sound   -- expect: no `sorryAx`
EOF
```

## Producing a certificate

```python
import discopt.modeling as dm
from discopt.certificate import build_feasibility_certificate, write_certificate

m = dm.Model()
x = m.continuous("x", lb=0, ub=4); y = m.continuous("y", lb=0, ub=4)
m.subject_to(x + y <= 5); m.subject_to(x * y >= 3)
m.minimize((x - 2) ** 2 + (y - 1) ** 2)
r = m.solve()
write_certificate(build_feasibility_certificate(m, r), "cert.json")
```

`python scripts/lean_certificate_demo.py` runs the whole round-trip (solve → emit
→ check, plus tamper-rejection) using the Python reference checker
`discopt.certificate.refcheck`, which is the executable twin of `checkFeasible`
and must agree with it on every accept/reject.

## Layout

| file | role |
|------|------|
| `Discopt/Model.lean` | model semantics, `checkFeasible`, `checkFeasible_sound` (trusted) |
| `Discopt/Json.lean`  | certificate JSON → `Certificate` (untrusted glue; a parse error rejects) |
| `Main.lean`          | the `check` executable |
