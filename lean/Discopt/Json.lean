/-
  Parse an emitted certificate JSON into `Discopt.Certificate`.

  This is *untrusted glue*: a parse error simply rejects the certificate, and the
  trusted checker (`Discopt.checkFeasible`) reasons only about the parsed value. It
  uses `Lean.Json` from the standard toolchain -- no external dependency.
-/
import Discopt.Model
import Lean.Data.Json

namespace Discopt
open Lean (Json)

abbrev PErr := Except String

/-- A `[num, den]` JSON pair -> exact `Rat`. -/
def parseRat (j : Json) : PErr Rat := do
  match (← j.getArr?).toList with
  | [jn, jd] =>
    let num ← jn.getInt?
    let den ← jd.getInt?
    if den ≤ 0 then throw s!"non-positive denominator {den}"
    return mkRat num den.toNat
  | l => throw s!"expected [num, den], got array of size {l.length}"

/-- Optional rational: JSON `null` -> `none`. -/
def parseRatOpt (j : Json) : PErr (Option Rat) :=
  match j with
  | Json.null => pure none
  | _ => (some ·) <$> parseRat j

def parseVType : String → PErr VType
  | "continuous" => pure .cont
  | "integer" => pure .int
  | "binary" => pure .bin
  | s => throw s!"unknown var type {s}"

def parseSense : String → PErr Sense
  | "le" => pure .le
  | "ge" => pure .ge
  | "eq" => pure .eq
  | s => throw s!"unknown constraint sense {s}"

def parseOSense : String → PErr OSense
  | "min" => pure .min
  | "max" => pure .max
  | s => throw s!"unknown objective sense {s}"

partial def parseExpr (j : Json) : PErr Expr := do
  let k ← (← j.getObjVal? "k").getStr?
  match k with
  | "const" => return .const (← parseRat (← j.getObjVal? "v"))
  | "var"   => return .var (← (← j.getObjVal? "i").getNat?)
  | "neg"   => return .neg (← parseExpr (← j.getObjVal? "x"))
  | "abs"   => return .abs (← parseExpr (← j.getObjVal? "x"))
  | "add"   => return .add (← parseExpr (← j.getObjVal? "l")) (← parseExpr (← j.getObjVal? "r"))
  | "sub"   => return .sub (← parseExpr (← j.getObjVal? "l")) (← parseExpr (← j.getObjVal? "r"))
  | "mul"   => return .mul (← parseExpr (← j.getObjVal? "l")) (← parseExpr (← j.getObjVal? "r"))
  | "div"   => return .div (← parseExpr (← j.getObjVal? "l")) (← parseExpr (← j.getObjVal? "r"))
  | "pow"   => return .pow (← parseExpr (← j.getObjVal? "l")) (← parseExpr (← j.getObjVal? "r"))
  | "fn"    =>
      let name ← (← j.getObjVal? "name").getStr?
      let args ← (← j.getObjVal? "args").getArr?
      return .fn name (← args.toList.mapM parseExpr)
  | other => throw s!"unknown expression node {other}"

def parseColumn (j : Json) : PErr Column := do
  return {
    name := (← (← j.getObjVal? "name").getStr?)
    vtype := (← parseVType (← (← j.getObjVal? "type").getStr?))
    lb := (← parseRatOpt (← j.getObjVal? "lb"))
    ub := (← parseRatOpt (← j.getObjVal? "ub"))
  }

def parseConstraint (j : Json) : PErr Constraint := do
  return {
    name := (← (← j.getObjVal? "name").getStr?)
    sense := (← parseSense (← (← j.getObjVal? "sense").getStr?))
    body := (← parseExpr (← j.getObjVal? "body"))
    rhs := (← parseRat (← j.getObjVal? "rhs"))
  }

def parseModel (j : Json) : PErr Model := do
  let cols ← (← j.getObjVal? "columns").getArr?
  let cons ← (← j.getObjVal? "constraints").getArr?
  let obj ← j.getObjVal? "objective"
  return {
    columns := (← cols.toList.mapM parseColumn)
    constraints := (← cons.toList.mapM parseConstraint)
    objective := {
      osense := (← parseOSense (← (← obj.getObjVal? "sense").getStr?))
      body := (← parseExpr (← obj.getObjVal? "body"))
    }
  }

/-- Parse the top-level document (the `certificate` object). -/
def parseCertificate (doc : Json) : PErr Certificate := do
  let c ← doc.getObjVal? "certificate"
  let tier ← (← c.getObjVal? "tier").getStr?
  if tier != "feasibility" then
    throw s!"unsupported tier {tier} (this checker is Tier 1)"
  let inc ← c.getObjVal? "incumbent"
  let xs ← (← inc.getObjVal? "x").getArr?
  let tol ← c.getObjVal? "tolerances"
  return {
    model := (← parseModel (← c.getObjVal? "model"))
    incumbent := (← xs.toList.mapM parseRat)
    objectiveValue := (← parseRat (← inc.getObjVal? "objectiveValue"))
    feasTol := (← parseRat (← tol.getObjVal? "feas"))
    intTol := (← parseRat (← tol.getObjVal? "int"))
  }

/-- Parse a certificate from raw JSON text. -/
def parseCertificateStr (s : String) : PErr Certificate := do
  parseCertificate (← Json.parse s)

end Discopt
