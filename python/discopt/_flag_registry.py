"""The index of every ``DISCOPT_*`` environment flag outside :class:`SolverTuning`.

The architecture review (2026-07-28 §2.4) found "no enumeration anywhere": 104 live
flags, zero reference documentation, and 12 daemon flags whose names are built by
f-string and are invisible to ``grep``. This module is that enumeration.

Scope. One row per flag that is **not** a :class:`~discopt.solver_tuning.SolverTuning`
field — those are typed dataclass fields and are documented from the dataclass
itself. Rust-side flags are included (they are part of the same user-facing
surface). ``docs/reference/flags.md`` is generated from this table plus the
``SolverTuning`` fields by ``scripts/gen_flag_docs.py``.

Kinds:

``graduated``
    Default-**ON** after a differential panel. Keeps its ``=0`` opt-out forever
    (CLAUDE.md §5).
``parked``
    Default-**OFF** opt-in: implemented, sound, awaiting (or failed) graduation.
``permanent``
    Infrastructure knob, not on the graduation track — budgets, paths, sockets,
    process lifecycle.
``debug``
    Developer instrumentation / entry-experiment lever. Never a shipped behavior.

Invariant enforced by ``python/tests/test_flag_registry.py``: every string-literal
flag name passed to :mod:`discopt._env` anywhere in ``python/discopt`` resolves to a
row here, and the daemon's f-string-built names resolve fully expanded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "FlagSpec",
    "FLAG_REGISTRY",
    "KINDS",
    "DAEMON_PREFIXES",
    "DAEMON_SUFFIXES",
    "solver_tuning_flags",
]

KINDS = ("graduated", "parked", "permanent", "debug")


@dataclass(frozen=True)
class FlagSpec:
    """One environment flag: what it is, what it defaults to, and who owns it."""

    name: str
    default: object
    kind: str
    issue: Optional[str]
    doc: str
    #: ``"python"`` or ``"rust"`` — which side reads it.
    side: str = "python"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} not in {KINDS}")
        if self.side not in ("python", "rust"):
            raise ValueError(f"{self.name}: side {self.side!r} not in ('python', 'rust')")


#: Env-var prefixes the daemon config expands (``_daemon_core.DaemonConfig``).
DAEMON_PREFIXES = ("DISCOPT_SOLVE", "DISCOPT_GAMS")
#: Suffixes appended to each prefix. ``BENCHMARK`` also shifts the other defaults.
DAEMON_SUFFIXES = (
    "BENCHMARK",
    "IDLE_TIMEOUT",
    "MAX_LIFETIME",
    "MAX_SOLVES",
    "MAX_RSS_MB",
    "JAX_CLEAR_EVERY",
)


def _daemon_rows() -> "list[FlagSpec]":
    """The 12 f-string-built daemon flags, expanded so they are greppable."""
    per_suffix = {
        "BENCHMARK": (
            False,
            "Benchmark mode: no solve cap, no lifetime cap, 30 min idle timeout.",
        ),
        "IDLE_TIMEOUT": (
            "600.0 (1800.0 under BENCHMARK)",
            "Seconds of idleness before the daemon exits.",
        ),
        "MAX_LIFETIME": (
            "3600.0 (0.0 under BENCHMARK)",
            "Seconds of total lifetime before the daemon recycles; 0 disables.",
        ),
        "MAX_SOLVES": (
            "500 (0 under BENCHMARK)",
            "Solves served before the daemon recycles; 0 disables.",
        ),
        "MAX_RSS_MB": (0, "Peak RSS (MiB) before the daemon recycles; 0 disables."),
        "JAX_CLEAR_EVERY": (
            0,
            "Drop JAX compilation caches every N solves; 0 disables.",
        ),
    }
    rows = []
    for prefix in DAEMON_PREFIXES:
        which = "solve" if prefix == "DISCOPT_SOLVE" else "GAMS-link"
        for suffix in DAEMON_SUFFIXES:
            default, doc = per_suffix[suffix]
            rows.append(
                FlagSpec(
                    name=f"{prefix}_{suffix}",
                    default=default,
                    kind="permanent",
                    issue=None,
                    doc=f"{which} daemon: {doc}",
                )
            )
    return rows


_ROWS: "list[FlagSpec]" = [
    # ------------------------------------------------------------------ solver.py
    FlagSpec(
        "DISCOPT_OBBT_TOPK",
        False,
        "parked",
        "T2.5",
        "Scored top-k per-node OBBT de-gate; awaiting the differential + panel gates.",
    ),
    FlagSpec(
        "DISCOPT_TRIVIAL_PRIMAL",
        False,
        "parked",
        "#827",
        "Seed the root with verified-feasible trivial points (origin, box centre, "
        "all-lb, all-ub) on pure-continuous models.",
    ),
    FlagSpec(
        "DISCOPT_QUBO_PRIMAL",
        True,
        "graduated",
        "#843",
        "Greedy-1opt + tabu local search seeding an incumbent on unconstrained "
        "binary quadratic models. Graduated 2026; `=0` restores the no-seed path.",
    ),
    FlagSpec(
        "DISCOPT_OBBT_ITERATE",
        False,
        "parked",
        "#282",
        "Iterate root OBBT to convergence instead of a fixed round budget.",
    ),
    FlagSpec(
        "DISCOPT_NATIVE_SPATIAL_KERNEL",
        False,
        "parked",
        "#764",
        "Route the spatial B&B tree into the native Rust kernel.",
    ),
    FlagSpec(
        "DISCOPT_NODE_PROBING",
        False,
        "parked",
        "cert:P3",
        "Per-node probing on discrete variables (sound: contracts only on proven "
        "infeasibility); costs O(discrete) extra FBBT solves per firing.",
    ),
    FlagSpec(
        "DISCOPT_NODE_PROBE_MAX_VARS",
        32,
        "permanent",
        "cert:P3",
        "Budget for `DISCOPT_NODE_PROBING`: discrete variables probed per firing.",
    ),
    FlagSpec(
        "DISCOPT_NLP_NATIVE",
        False,
        "parked",
        None,
        "Use the native (pyo3) NLP problem object instead of the Python evaluator; "
        "blocked on `PyNlProblem` being Send-safe.",
    ),
    FlagSpec(
        "DISCOPT_SEPARATION_LP_SIMPLEX",
        True,
        "graduated",
        None,
        "Solve separation / strong-branching LPs with the in-house warm simplex "
        "instead of a cold POUNCE IPM solve. `=0` restores the caller's backend.",
    ),
    FlagSpec(
        "DISCOPT_ROOT_CUT_ROUNDS",
        0,
        "permanent",
        None,
        "Default for `solve_model(root_cut_rounds=...)`: root cut-pool rounds.",
    ),
    FlagSpec(
        "DISCOPT_ROOT_CUT_MAX",
        200,
        "permanent",
        None,
        "Default for `solve_model(root_cut_max=...)`: root cut-pool size cap.",
    ),
    FlagSpec(
        "DISCOPT_CMIR_AGGREGATION",
        False,
        "parked",
        "cert:P3",
        "Marchand-Wolsey aggregation c-MIR separator. Validity-gated, so enabling "
        "it can only add valid cuts.",
    ),
    FlagSpec(
        "DISCOPT_P3_FORCE_CUT_PATH",
        False,
        "debug",
        "cert:P3.1c",
        "Entry-experiment lever: skip the big-M `nlp_solver -> 'simplex'` reroute so "
        "the integer-product class stays on the cut-bearing `_solve_milp_bb` path.",
    ),
    FlagSpec(
        "DISCOPT_GP_MINLP",
        False,
        "parked",
        None,
        "Geometric-programming MINLP fast path (`discopt.gp.solve_gp_minlp`).",
    ),
    FlagSpec(
        "DISCOPT_SGO",
        False,
        "parked",
        "#114/#741",
        "Signomial global fast path: spatial B&B on the certified log-domain DC "
        "envelope for mixed-sign signomials over a positive box.",
    ),
    FlagSpec(
        "DISCOPT_DECOMP_STORE",
        None,
        "permanent",
        None,
        "Path to the decomposition-advisor outcome store (`RecordStore`).",
    ),
    FlagSpec(
        "DISCOPT_HEUR_BUDGET",
        True,
        "graduated",
        "#347",
        "SCIP-shaped success-weighted contingent gating the heavy primal improvers "
        "(enumeration, RINS, local branching). `=0` restores always-on.",
    ),
    FlagSpec(
        "DISCOPT_HEUR_OFFSET",
        0.0,
        "permanent",
        "#347",
        "Root contingent (sub-NLP-solve-equivalents) for `DISCOPT_HEUR_BUDGET`.",
    ),
    FlagSpec(
        "DISCOPT_HEUR_QUOT",
        0.5,
        "permanent",
        "#347",
        "Per-processed-node contingent accrual for `DISCOPT_HEUR_BUDGET`.",
    ),
    FlagSpec(
        "DISCOPT_NARROW_BOX_BRANCH",
        False,
        "parked",
        "#732",
        "Convert a failed *branchable* narrow-box node into an open node instead of "
        "a sentinel fathom.",
    ),
    FlagSpec(
        "DISCOPT_ROOT_BUDGET_GATE",
        True,
        "graduated",
        None,
        "Refuse to launch a root heuristic NLP that cannot fit in the remaining "
        "wall budget. Primal-only, so skipping is always sound.",
    ),
    FlagSpec(
        "DISCOPT_ROOT_FIXPOINT_REPOOL",
        False,
        "parked",
        None,
        "Re-separate the root cut pool after a root-fixpoint bound tightening.",
    ),
    FlagSpec(
        "DISCOPT_MILP_SWAP_RESEED",
        True,
        "graduated",
        None,
        "One-hot swap reseeding of the MILP incumbent on re-entry (graphpart "
        "family). `=0` is the opt-out.",
    ),
    # --------------------------------------------------------- package / process
    FlagSpec(
        "DISCOPT_DISABLE_JAX_CACHE",
        False,
        "permanent",
        None,
        "Skip enabling JAX's persistent on-disk compilation cache at import.",
    ),
    FlagSpec(
        "DISCOPT_EAGER_IMPORTS",
        False,
        "permanent",
        None,
        "Import the whole solve path at `import discopt` instead of lazily.",
    ),
    FlagSpec(
        "DISCOPT_LLM_MODEL",
        None,
        "permanent",
        None,
        "litellm model id for the optional LLM features (`discopt.llm`).",
    ),
    FlagSpec(
        "DISCOPT_GAMS_NO_DAEMON",
        False,
        "permanent",
        None,
        "GAMS link: solve in-process instead of via the warm daemon.",
    ),
    FlagSpec(
        "DISCOPT_SOLVE_SOCKET",
        None,
        "permanent",
        None,
        "Explicit AF_UNIX socket path for the solve daemon.",
    ),
    FlagSpec(
        "DISCOPT_GAMS_SOCKET",
        None,
        "permanent",
        None,
        "Explicit AF_UNIX socket path for the GAMS-link daemon.",
    ),
    # ---------------------------------------------------------------- heuristics
    FlagSpec(
        "DISCOPT_HEURISTIC_GOVERNOR",
        True,
        "graduated",
        "G2",
        "Hit-rate-adaptive governor throttling primal heuristic sources. `=0` "
        "restores the pre-governor byte-identical behaviour.",
    ),
    # ------------------------------------------------------------------ presolve
    FlagSpec(
        "DISCOPT_PRESOLVE_SUBSTITUTE",
        False,
        "parked",
        None,
        "Solve from the presolved representation (substitution + postsolve chain) "
        "rather than copying bounds only.",
    ),
    FlagSpec(
        "DISCOPT_COEF_TIGHTEN",
        False,
        "parked",
        None,
        "Python coefficient strengthening at the root. Sound (the LP relaxation "
        "only shrinks) but not yet panel-graduated.",
    ),
    FlagSpec(
        "DISCOPT_NLPBB_ROOT_CUTS",
        False,
        "parked",
        None,
        "Root cut loop on the NLP-B&B path.",
    ),
    # NOTE: ``DISCOPT_OBBT_CASCADE_AUX`` used to live here. Consolidation-plan
    # Card 2a moved it into ``SolverTuning.obbt_cascade_aux`` so it is resolved
    # ONCE for all six ``obbt_tighten_root`` call sites instead of at the single
    # ``root_reduce`` site. It is documented by ``solver_tuning_flags()``; listing
    # it in both places would trip ``test_registry_and_solver_tuning_do_not_overlap``.
    # ------------------------------------------------------------- convex kernel
    FlagSpec(
        "DISCOPT_CONVEX_KERNEL",
        False,
        "parked",
        "#798/#779",
        "Route certifiable convex MINLPs into the native convex kernel; the result "
        "is adopted only when it certifies optimality and verifies feasible.",
    ),
    FlagSpec(
        "DISCOPT_CVX_DOMINATED_COLS",
        True,
        "graduated",
        "#879",
        "Dominated-cost-column upper bound inside the convex kernel.",
    ),
    FlagSpec(
        "DISCOPT_CONVEX_KERNEL_BUDGET",
        120.0,
        "permanent",
        "#798",
        "Wall-clock budget (s) for a convex-kernel attempt before falling back.",
    ),
    FlagSpec(
        "DISCOPT_FBBT_SEED",
        False,
        "parked",
        "consolidation-plan Card 3e",
        "Seed the Rust root FBBT pass from the presolve orchestrator's running box "
        "instead of only the model's declared box (read Python-side in "
        "`_jax/presolve_pipeline.run_root_presolve`, forwarded as the "
        "`fbbt_seed_from_ctx` kwarg of `PyModelRepr.presolve`). Without it the "
        "wired-in `fbbt` pass re-derives the declared box on every sweep and can "
        "compose with NO other pass's tightenings (measured: 0 composed bounds "
        "across 7/7 instances, against `fbbt_fp`'s 48). Bound-changing (strictly "
        "tightening), so it is default-OFF until a differential panel graduates it.",
    ),
    # ------------------------------------------------------------- reformulation
    FlagSpec(
        "DISCOPT_LIFT_ZERO_SPANNING_FACTORS",
        True,
        "graduated",
        None,
        "Lift zero-spanning product factors during factorable reformulation. `=0` "
        "restores the byte-identical no-tagging behaviour.",
    ),
    FlagSpec(
        "DISCOPT_LIFT_LOOSE_PRODUCTS",
        True,
        "graduated",
        "TD-A/T2.6",
        "Lift integer powers of transcendental univariates into `t == g(x)` auxiliaries.",
    ),
    FlagSpec(
        "DISCOPT_INTEGER_RATIO_PARTITION",
        True,
        "graduated",
        None,
        "Integer-ratio partitioning reformulation (graduated 2026-07-16 on a "
        "66-instance differential panel).",
    ),
    FlagSpec(
        "DISCOPT_MULTILINEAR_COUPLING_RLT",
        False,
        "parked",
        "#721",
        "Objective-coupling RLT on top of the integer-multilinear reformulation.",
    ),
    # ------------------------------------------------------------ relaxation/LP
    FlagSpec(
        "DISCOPT_REDUCED_LP_BACKEND",
        "simplex",
        "debug",
        None,
        "Backend for the reduced-space McCormick Kelley LP: `simplex` (default) or `scipy`.",
    ),
    FlagSpec(
        "DISCOPT_ANALYTIC_SEPGRAD",
        False,
        "parked",
        None,
        "Use the compiled analytic separation gradient instead of the JAX path "
        "(falls back to JAX on any construction failure).",
    ),
    FlagSpec(
        "DISCOPT_LOGSUMEXP_ATOM",
        False,
        "parked",
        None,
        "Emit the convex softmax-tangent OA for `log(sum exp(.))` instead of the "
        "loose concave `log` relaxation.",
    ),
    FlagSpec(
        "DISCOPT_NORM_ATOM",
        False,
        "parked",
        None,
        "Emit the convex OA of `sqrt(sum t^2)` instead of the loose concave sqrt.",
    ),
    FlagSpec(
        "DISCOPT_ENTROPY_ATOM",
        False,
        "parked",
        None,
        "Recognize `x*log(x)` on `x>0` and emit its exact 1-D convex envelope.",
    ),
    FlagSpec(
        "DISCOPT_XEXP_ATOM",
        False,
        "parked",
        None,
        "Recognize `t*exp(t)` on its convex region `t>=-2` and emit the exact 1-D convex envelope.",
    ),
    FlagSpec(
        "DISCOPT_RELENT_ATOM",
        False,
        "parked",
        None,
        "Jointly-convex OA of the relative entropy `x*log(x/y)`.",
    ),
    FlagSpec(
        "DISCOPT_INCREMENTAL_MC",
        True,
        "graduated",
        "cert:T1.3",
        "Build the incremental McCormick LP (row-for-row self-validated) instead of "
        "cold-building the relaxation per node.",
    ),
    FlagSpec(
        "DISCOPT_PSD_QFORM",
        False,
        "parked",
        None,
        "PSD quadratic-form convexity certificate.",
    ),
    FlagSpec(
        "DISCOPT_G_CONVEX_CUTS",
        False,
        "parked",
        None,
        "Inject cuts derived from g-convexity certificates.",
    ),
    # -------------------------------------------------------------- spatial B&B
    FlagSpec(
        "DISCOPT_LP_SPATIAL_PLUNGE",
        "require_incremental",
        "permanent",
        None,
        "Depth-first plunging in the LP spatial B&B loop; unset defers to the "
        "caller's `require_incremental`.",
    ),
    FlagSpec(
        "DISCOPT_LP_SPATIAL_FALLBACK",
        True,
        "graduated",
        None,
        "Allow the LP-spatial path to fall back to the generic spatial loop.",
    ),
    FlagSpec(
        "DISCOPT_LP_SPATIAL_MIXED",
        False,
        "parked",
        "#860",
        "Extend the LP-spatial fallback to mixed integer/continuous models.",
    ),
    # -------------------------------------------------------------- benchmarking
    FlagSpec(
        "DISCOPT_MINLP_BENCH",
        None,
        "permanent",
        None,
        "Path to the MINLPLib snapshot used by the benchmark harness and the corpus-drawing tests.",
    ),
    # ---------------------------------------------------------------------- Rust
    FlagSpec(
        "DISCOPT_PROFILE",
        False,
        "debug",
        None,
        "Enable the Rust core's internal phase profiler.",
        side="rust",
    ),
    FlagSpec(
        "DISCOPT_DISABLE_CSE",
        False,
        "debug",
        None,
        "Build the `.nl` expression arena with plain append instead of interning "
        "(evaluation-identical; quantifies the CSE node-count lever).",
        side="rust",
    ),
    FlagSpec(
        "DISCOPT_T14_DBG",
        False,
        "debug",
        "T14",
        "Print the warm-basis accept/reject decision in the primal simplex.",
        side="rust",
    ),
    FlagSpec(
        "DISCOPT_CVX_NATIVELP",
        False,
        "parked",
        "#807",
        "Route convex-kernel node solves through the shared persistent LP "
        "(bounds-in-place dual-warm reoptimize) instead of a cold per-node solve.",
        side="rust",
    ),
    FlagSpec(
        "DISCOPT_LU_DENSITY_ROUTE",
        True,
        "graduated",
        "#602/#612",
        "Density-based routing between the sparse and dense LU factorizations. "
        "`=0` restores the historical dense-preferring routing byte-identically.",
        side="rust",
    ),
    FlagSpec(
        "DISCOPT_LP_FACTORIZATION_HARDENING",
        False,
        "parked",
        "#671",
        "Failure-triggered hardened retry: build the basis factor with a singular "
        "perturbation so a near-singular basis completes.",
        side="rust",
    ),
] + _daemon_rows()

#: Every non-``SolverTuning`` flag, keyed by name.
FLAG_REGISTRY: "dict[str, FlagSpec]" = {row.name: row for row in _ROWS}

if len(FLAG_REGISTRY) != len(_ROWS):  # pragma: no cover - guards a copy/paste slip
    _dupes = sorted({r.name for r in _ROWS if sum(1 for x in _ROWS if x.name == r.name) > 1})
    raise RuntimeError(f"duplicate FLAG_REGISTRY rows: {_dupes}")


def solver_tuning_flags() -> "dict[str, tuple[str, object, str]]":
    """The ``SolverTuning`` half of the flag surface: ``{env name: (field, default, doc)}``.

    The mapping lives inside the fields' ``default_factory`` lambdas, so it is
    recovered by parsing the dataclass rather than duplicated by hand — a hand-copy
    is exactly the kind of drift this Phase-1 card exists to remove. Shared by
    ``scripts/gen_flag_docs.py`` and ``test_flag_registry.py``.

    Values are ``(field name, the default a bare ``SolverTuning()`` resolves to,
    the field's attribute docstring)``.
    """
    import ast
    import inspect

    from discopt import solver_tuning as _st

    tree = ast.parse(inspect.getsource(_st))

    def literals(node: ast.AST) -> "list[str]":
        return [
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("DISCOPT_")
        ]

    # Module-level helpers (``_env_trilinear``) hide the literal behind a call.
    helper_flags: "dict[str, list[str]]" = {}
    class_node: "ast.ClassDef | None" = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            helper_flags[node.name] = literals(node)
        elif isinstance(node, ast.ClassDef) and node.name == "SolverTuning":
            class_node = node
    if class_node is None:  # pragma: no cover - the class is the module's reason to exist
        raise RuntimeError("SolverTuning class not found in discopt.solver_tuning")

    blank = _st.SolverTuning()
    out: "dict[str, tuple[str, object, str]]" = {}
    body = class_node.body
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        field_name = stmt.target.id
        names = list(literals(stmt)) if stmt.value is not None else []
        if stmt.value is not None:
            for call in ast.walk(stmt.value):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    names.extend(helper_flags.get(call.func.id, []))
        if not names:
            continue
        doc = ""
        if i + 1 < len(body):
            nxt = body[i + 1]
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                doc = " ".join(nxt.value.value.split())
        for env_name in dict.fromkeys(names):
            out.setdefault(env_name, (field_name, getattr(blank, field_name), doc))
    return out
