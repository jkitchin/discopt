/-
  discopt Tier-1 feasibility certificate: model, semantics, checker, soundness.

  Pure Lean 4 **core** (no Mathlib): a feasibility certificate is checked over
  exact rationals (`Rat`), for which core provides arithmetic and a decidable
  order -- all the Tier-1 checker needs. Transcendental `MathFunc`s (`Expr.fn`)
  are not evaluable in exact `Rat`; the checker conservatively fails on them
  (Phase 1 lifts them to Mathlib reals with interval enclosures -- see
  docs/dev/lean-certificate-plan.md).

  Trust story: `Feasible` and `ObjOk` are *semantic* propositions (real `≤` over
  `Rat`, `eval = some v`). They are decidable, so `checkFeasible` is literally
  `decide (Feasible ∧ ObjOk)` and `checkFeasible_sound` is `of_decide_eq_true` --
  there is no gap between what is checked and what is proved. Per certificate the
  kernel only *evaluates* the decision procedure; it never searches for a proof.

  The Python reference checker (`python/discopt/certificate/refcheck.py`) is the
  executable twin of this file and must agree on every accept/reject.
-/

namespace Discopt

/-- Constraint relation (each holds within a tolerance). -/
inductive Sense where
  | le | ge | eq
  deriving Repr, DecidableEq

/-- Objective direction. -/
inductive OSense where
  | min | max
  deriving Repr, DecidableEq

/-- Variable domain; `int`/`bin` carry an integrality obligation. -/
inductive VType where
  | cont | int | bin
  deriving Repr, DecidableEq

/-- Expression DAG over flat columns; mirrors the emitter's tagged encoding. -/
inductive Expr where
  | const : Rat → Expr
  | var   : Nat → Expr
  | neg   : Expr → Expr
  | abs   : Expr → Expr
  | add   : Expr → Expr → Expr
  | sub   : Expr → Expr → Expr
  | mul   : Expr → Expr → Expr
  | div   : Expr → Expr → Expr
  | pow   : Expr → Expr → Expr
  | fn    : String → List Expr → Expr
  deriving Repr

/-- `Int → Rat` without relying on an `IntCast` instance (core-only). -/
def ratOfInt (n : Int) : Rat := mkRat n 1

/-- `|r|` for a rational (core provides no `abs` on `Rat`). -/
def ratAbs (r : Rat) : Rat := if r < 0 then -r else r

/-- `base ^ n` for a natural exponent. -/
def natPow (base : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => base * natPow base n

/-- `base ^ e` for an integer exponent; `none` on `0 ^ (negative)`. -/
def ratPow (base : Rat) : Int → Option Rat
  | Int.ofNat n => some (natPow base n)
  | Int.negSucc n => if base == 0 then none else some (natPow (1 / base) (n + 1))

/-- Evaluate an expression at incumbent `xs`. `none` = "not evaluable in exact
    `Rat`" (bad column index, division by zero, non-integer power, or a
    transcendental `fn`); the checker treats an unevaluable body as unverified. -/
def Expr.eval (xs : List Rat) : Expr → Option Rat
  | .const c => some c
  | .var i => xs[i]?
  | .neg a => (a.eval xs).map (fun v => -v)
  | .abs a => (a.eval xs).map ratAbs
  | .add a b => do return (← a.eval xs) + (← b.eval xs)
  | .sub a b => do return (← a.eval xs) - (← b.eval xs)
  | .mul a b => do return (← a.eval xs) * (← b.eval xs)
  | .div a b => do
      let bv ← b.eval xs
      if bv == 0 then none else return (← a.eval xs) / bv
  | .pow a b => do
      let bv ← b.eval xs
      if bv.den == 1 then ratPow (← a.eval xs) bv.num else none
  | .fn _ _ => none

/-- A flattened scalar column: domain type and (optional) finite bounds. -/
structure Column where
  name  : String
  vtype : VType
  lb    : Option Rat
  ub    : Option Rat
  deriving Repr

/-- A single algebraic constraint `body ⋈ rhs`. -/
structure Constraint where
  name  : String
  sense : Sense
  body  : Expr
  rhs   : Rat
  deriving Repr

structure Objective where
  osense : OSense
  body   : Expr
  deriving Repr

structure Model where
  columns     : List Column
  constraints : List Constraint
  objective   : Objective
  deriving Repr

structure Certificate where
  model          : Model
  incumbent      : List Rat
  objectiveValue : Rat
  feasTol        : Rat
  intTol         : Rat
  deriving Repr

/-! ### Semantic feasibility (the meaning we prove)

Every predicate is written as a `match` on an `Option`/enum rather than a
`∀`-implication, so its `Decidable` instance synthesizes structurally. -/

/-- Value `v` satisfies the `sense`-relation to `rhs` within `tol`. -/
def senseHolds (s : Sense) (v rhs tol : Rat) : Prop :=
  match s with
  | .le => v ≤ rhs + tol
  | .ge => rhs - tol ≤ v
  | .eq => ratAbs (v - rhs) ≤ tol

instance (s : Sense) (v rhs tol : Rat) : Decidable (senseHolds s v rhs tol) := by
  unfold senseHolds; cases s <;> infer_instance

/-- `v` respects a lower bound (vacuous when the bound is open). -/
def lbHolds (tol : Rat) (lb : Option Rat) (v : Rat) : Prop :=
  match lb with
  | some lo => lo - tol ≤ v
  | none => True

instance (tol : Rat) (lb : Option Rat) (v : Rat) : Decidable (lbHolds tol lb v) := by
  unfold lbHolds; cases lb <;> infer_instance

/-- `v` respects an upper bound (vacuous when the bound is open). -/
def ubHolds (tol : Rat) (ub : Option Rat) (v : Rat) : Prop :=
  match ub with
  | some hi => v ≤ hi + tol
  | none => True

instance (tol : Rat) (ub : Option Rat) (v : Rat) : Decidable (ubHolds tol ub v) := by
  unfold ubHolds; cases ub <;> infer_instance

/-- `v` lies within the column's (possibly open) bounds, up to `tol`. -/
def withinBounds (tol : Rat) (c : Column) (v : Rat) : Prop :=
  lbHolds tol c.lb v ∧ ubHolds tol c.ub v

instance (tol : Rat) (c : Column) (v : Rat) : Decidable (withinBounds tol c v) := by
  unfold withinBounds; infer_instance

/-- `v` is within `tol` of its nearest integer. (If any integer is within `tol`,
    the nearest one is; so this is exactly "within `tol` of an integer".) -/
def roundRat (v : Rat) : Int := Int.fdiv (2 * v.num + (v.den : Int)) (2 * (v.den : Int))

def isIntegral (tol v : Rat) : Prop := ratAbs (v - ratOfInt (roundRat v)) ≤ tol

instance (tol v : Rat) : Decidable (isIntegral tol v) := by
  unfold isIntegral; infer_instance

/-- Integrality obligation by domain type (vacuous for continuous). -/
def integralHolds (intTol : Rat) (t : VType) (v : Rat) : Prop :=
  match t with
  | .cont => True
  | .int => isIntegral intTol v
  | .bin => isIntegral intTol v

instance (intTol : Rat) (t : VType) (v : Rat) : Decidable (integralHolds intTol t v) := by
  unfold integralHolds; cases t <;> infer_instance

/-- Per-column obligation: bounds always, integrality on int/bin columns. -/
def colHolds (feasTol intTol : Rat) (cv : Column × Rat) : Prop :=
  withinBounds feasTol cv.1 cv.2 ∧ integralHolds intTol cv.1.vtype cv.2

instance (feasTol intTol : Rat) (cv : Column × Rat) : Decidable (colHolds feasTol intTol cv) := by
  unfold colHolds; infer_instance

/-- A constraint holds: its body evaluates and the value satisfies the relation. -/
def consHolds (xs : List Rat) (tol : Rat) (c : Constraint) : Prop :=
  match c.body.eval xs with
  | some v => senseHolds c.sense v c.rhs tol
  | none => False

instance (xs : List Rat) (tol : Rat) (c : Constraint) : Decidable (consHolds xs tol c) := by
  unfold consHolds; cases c.body.eval xs <;> infer_instance

/-- Full feasibility of the incumbent against the model. -/
def Feasible (cert : Certificate) : Prop :=
  cert.incumbent.length = cert.model.columns.length ∧
  (∀ cv ∈ cert.model.columns.zip cert.incumbent, colHolds cert.feasTol cert.intTol cv) ∧
  (∀ c ∈ cert.model.constraints, consHolds cert.incumbent cert.feasTol c)

instance (cert : Certificate) : Decidable (Feasible cert) := by
  unfold Feasible; infer_instance

/-- The reported objective value is the true value at the incumbent (within tol). -/
def ObjOk (cert : Certificate) : Prop :=
  match cert.model.objective.body.eval cert.incumbent with
  | some v => ratAbs (v - cert.objectiveValue) ≤ cert.feasTol
  | none => False

instance (cert : Certificate) : Decidable (ObjOk cert) := by
  unfold ObjOk; cases cert.model.objective.body.eval cert.incumbent <;> infer_instance

/-! ### The checker and its soundness -/

/-- The Tier-1 decision procedure. -/
def checkFeasible (cert : Certificate) : Bool := decide (Feasible cert ∧ ObjOk cert)

/-- **Soundness.** A `true` verdict is backed by a kernel proof that the incumbent
    is feasible and attains the reported objective value. The Tier-1 guarantee is
    an honest primal bound; Tiers 2/3 add the dual-bound / global-optimality
    conjuncts. -/
theorem checkFeasible_sound (cert : Certificate) :
    checkFeasible cert = true → Feasible cert ∧ ObjOk cert := fun h => of_decide_eq_true h

theorem checkFeasible_feasible (cert : Certificate) (h : checkFeasible cert = true) :
    Feasible cert := (checkFeasible_sound cert h).1

theorem checkFeasible_objective (cert : Certificate) (h : checkFeasible cert = true) :
    ObjOk cert := (checkFeasible_sound cert h).2

end Discopt
