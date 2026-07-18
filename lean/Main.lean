/-
  `check` executable: verify a discopt Tier-1 feasibility certificate.

  Usage:  lake exe check path/to/cert.json

  Exit code 0 and prints `FEASIBLE` when the incumbent is verified feasible with
  the reported objective value; exit code 1 and `REJECTED` otherwise (either the
  checker refuted the certificate, or the JSON could not be parsed).

  The guarantee, discharged by `Discopt.checkFeasible_sound`, is that a `FEASIBLE`
  verdict is backed by a kernel-checked proof of `Feasible ∧ ObjOk`.
-/
import Discopt.Model
import Discopt.Json

open Discopt

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
    let text ← IO.FS.readFile path
    match parseCertificateStr text with
    | .error e =>
      IO.eprintln s!"REJECTED: could not parse certificate: {e}"
      return 1
    | .ok cert =>
      if checkFeasible cert then
        IO.println "FEASIBLE: incumbent verified feasible with reported objective value"
        return 0
      else
        IO.eprintln "REJECTED: certificate does not verify"
        return 1
  | _ =>
    IO.eprintln "usage: check <certificate.json>"
    return 2
