import Lake
open Lake DSL

/-- discopt certificate checker.

    Deliberately depends only on the Lean 4 core toolchain and its bundled
    `Lean.Data.Json` -- **no Mathlib** -- so the Tier-1 feasibility checker builds
    fast and its trusted base is just the Lean kernel. Later tiers (interval
    enclosures over reals, McCormick envelopes, LP duality) will add a Mathlib
    dependency. -/
package «discopt_cert»

@[default_target]
lean_lib «Discopt»

@[default_target]
lean_exe «check» where
  root := `Main
