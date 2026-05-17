"""Streamlit app for ``discopt doe``.

Launched by ``discopt doe gui [WORKBOOK]``. Reads the target workbook
path from ``$DISCOPT_DOE_WORKBOOK`` if set; otherwise the user picks
or creates one from the sidebar.

The app is a UI shell over the same pure ``do_*`` functions the CLI
uses. Every interaction reads campaign state through ``do_status`` so
buttons enable themselves only when their preconditions hold —
making the UI robust to partially-filled response columns, missing
fit results, or a workbook that was edited in Excel underneath us.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import streamlit as st

from discopt.doe.cli import (
    DoEError,
    ExtendParams,
    NewParams,
    do_extend,
    do_fit,
    do_new,
    do_status,
)
from discopt.doe.workbook import SHEET_HISTORY, SHEET_RUNS

_WORKBOOK_ENV = "DISCOPT_DOE_WORKBOOK"


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _init_state() -> None:
    if "workbook_path" not in st.session_state:
        env_path = os.environ.get(_WORKBOOK_ENV, "").strip()
        if env_path and Path(env_path).is_file():
            st.session_state["workbook_path"] = env_path
        else:
            st.session_state["workbook_path"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None


def _set_workbook(path: str | Path | None) -> None:
    st.session_state["workbook_path"] = str(path) if path else None
    st.session_state["last_error"] = None


def _safe_status(path: str) -> dict[str, Any] | None:
    try:
        return do_status({"workbook": path})
    except (FileNotFoundError, ValueError, DoEError) as e:
        st.session_state["last_error"] = str(e)
        return None


def _runs_df(path: str) -> pd.DataFrame:
    """Read the full runs sheet as a DataFrame for the editable table."""
    wb = openpyxl.load_workbook(path)
    sheet = wb[SHEET_RUNS]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    headers = [str(c) for c in rows[0]]
    data = [r for r in rows[1:] if r and r[0] is not None]
    return pd.DataFrame(data, columns=headers)


def _save_responses(path: str, edited: pd.DataFrame, response: str) -> int:
    """Write edited response values back. Returns count of cells updated."""
    wb = openpyxl.load_workbook(path)
    sheet = wb[SHEET_RUNS]
    headers = [c.value for c in sheet[1]]
    resp_idx = headers.index(response)
    id_idx = headers.index("run_id")
    edits_by_id = {
        int(row["run_id"]): row[response]
        for _, row in edited.iterrows()
        if pd.notna(row["run_id"])
    }
    n_updates = 0
    for row in sheet.iter_rows(min_row=2):
        if row[id_idx].value is None:
            continue
        rid = int(row[id_idx].value)
        new_val = edits_by_id.get(rid)
        old_val = row[resp_idx].value
        if pd.isna(new_val) if new_val is not None else True:
            new_val = None
        if new_val != old_val:
            row[resp_idx].value = float(new_val) if new_val is not None else None
            n_updates += 1
    wb.save(path)
    return n_updates


def _history_rows(path: str) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(path)
    if SHEET_HISTORY not in wb.sheetnames:
        return []
    sheet = wb[SHEET_HISTORY]
    rows = list(sheet.iter_rows(values_only=True))
    if len(rows) < 2:
        return []
    headers = [str(c) for c in rows[0]]
    return [dict(zip(headers, r)) for r in rows[1:] if r and r[0] is not None]


# ──────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────


# GUI template menu collapses response-surface-2d and -3d into a single
# "response surface" choice; we pick the underlying template by factor
# count when generating.
_GUI_TEMPLATE_CHOICES = ("linear", "polynomial-1d", "response-surface")
_GUI_TEMPLATE_HELP = {
    "linear": (
        "First-order model: y = b0 + b1·x1 + ... + bn·xn. "
        "Use for screening — does the response depend on each factor "
        "at all? Works with any number of factors."
    ),
    "polynomial-1d": (
        "Single-factor polynomial: y = b0 + b1·x + b2·x² + ... + bd·xᵈ. "
        "Use when you have one factor and need to capture curvature. "
        "Pick the degree below."
    ),
    "response-surface": (
        "Full-quadratic response surface in 2 or 3 factors: intercept, "
        "main effects, pure quadratic terms, and pairwise interactions. "
        "Use for optimization once you know the factors matter."
    ),
}


def _sidebar() -> None:
    st.sidebar.title("discopt doe")

    mode_options = ["Open existing", "New campaign"]
    default_idx = 0 if st.session_state.get("workbook_path") else 1
    mode = st.sidebar.radio(
        "Workbook",
        mode_options,
        index=default_idx,
        help=(
            "**Open existing** loads an `.xlsx` campaign that's already "
            "on disk. **New campaign** generates a fresh optimal design "
            "and writes a new workbook."
        ),
    )

    if mode == "Open existing":
        _sidebar_open()
    else:
        _sidebar_new()


def _sidebar_open() -> None:
    current = st.session_state.get("workbook_path") or ""
    path_input = st.sidebar.text_input(
        "Path to .xlsx",
        value=current,
        help=(
            "Absolute or `~`-relative path to a `discopt doe` workbook. "
            "The file must already exist."
        ),
    )
    if st.sidebar.button(
        "Open",
        help="Load the workbook at the path above into the GUI.",
    ):
        candidate = Path(path_input).expanduser().resolve()
        if not candidate.is_file():
            st.sidebar.error(f"Not found: {candidate}")
        else:
            _set_workbook(candidate)
            st.rerun()

    uploaded = st.sidebar.file_uploader(
        "...or upload one",
        type="xlsx",
        key="uploader",
        help=(
            "Upload a workbook from your computer. A copy is saved to a "
            "temp directory; use the **Download workbook** button on the "
            "main page to grab it back when done."
        ),
    )
    if uploaded is not None and st.sidebar.button(
        "Use uploaded copy",
        help="Open the uploaded workbook in the GUI.",
    ):
        tmp = Path(tempfile.gettempdir()) / f"discopt-doe-{uploaded.name}"
        tmp.write_bytes(uploaded.getvalue())
        _set_workbook(tmp)
        st.rerun()


def _sidebar_new() -> None:
    template_choice = st.sidebar.selectbox(
        "Model template",
        _GUI_TEMPLATE_CHOICES,
        help=(
            "The model the design will be optimal for. Hover the option "
            "and read the caption below for what each template means."
        ),
    )
    st.sidebar.caption(_GUI_TEMPLATE_HELP[template_choice])

    if template_choice == "polynomial-1d":
        degree = st.sidebar.number_input(
            "Polynomial degree",
            min_value=1,
            max_value=10,
            value=2,
            help=(
                "Highest power of x in the model. Degree 1 = linear, "
                "2 = quadratic, 3 = cubic. The fit has `degree + 1` "
                "coefficients."
            ),
        )
        n_factors_default = 1
        factors_fixed = True
    elif template_choice == "response-surface":
        degree = None
        n_factors_default = 2
        factors_fixed = False
    else:  # linear
        degree = None
        n_factors_default = 2
        factors_fixed = False

    st.sidebar.markdown("**Factors**")
    if template_choice == "response-surface":
        st.sidebar.caption(
            "Add exactly 2 or 3 rows — one per design factor. Each needs "
            "a name and lower/upper bounds defining its experimental range."
        )
    elif template_choice == "linear":
        st.sidebar.caption(
            "One row per design factor. Add as many as you want; each needs "
            "a name and lower/upper bounds."
        )
    else:
        st.sidebar.caption(
            "One row for the single design factor: name plus the range "
            "the polynomial will be fit over."
        )
    default_factors = pd.DataFrame(
        {
            "name": [f"x{i + 1}" for i in range(n_factors_default)],
            "lb": [0.0] * n_factors_default,
            "ub": [1.0] * n_factors_default,
        }
    )
    num_rows = "fixed" if factors_fixed else "dynamic"
    factors_df = st.sidebar.data_editor(
        default_factors,
        num_rows=num_rows,
        key=f"factors_editor_{template_choice}",
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn(
                "name", help="Factor name; becomes a column header in the workbook."
            ),
            "lb": st.column_config.NumberColumn(
                "lb", help="Lower bound of the factor's experimental range."
            ),
            "ub": st.column_config.NumberColumn(
                "ub", help="Upper bound (must exceed lb)."
            ),
        },
    )

    response_name = st.sidebar.text_input(
        "Response name",
        value="y",
        help=(
            "Name of the measured quantity (e.g. 'yield', 'conversion'). "
            "Becomes the response column header in the workbook."
        ),
    )
    n = st.sidebar.number_input(
        "Initial runs",
        min_value=1,
        value=6,
        help=(
            "How many experiments to generate in this first batch. Need "
            "at least as many runs as parameters for a well-posed fit."
        ),
    )
    error = st.sidebar.number_input(
        "Measurement noise σ",
        min_value=1e-9,
        value=1.0,
        format="%g",
        help=(
            "Estimated standard deviation of the measurement error in the "
            "response, in the same units as the response. Used to weight "
            "the Fisher Information Matrix."
        ),
    )
    criterion = st.sidebar.selectbox(
        "Optimality criterion",
        ["determinant", "trace", "min_eigenvalue", "condition_number"],
        help=(
            "**determinant** (D-optimal, default) maximizes the FIM's "
            "log-determinant — minimizes confidence-ellipsoid volume. "
            "**trace** (A) minimizes average parameter variance. "
            "**min_eigenvalue** (E) maximizes worst-case parameter info. "
            "**condition_number** balances information across directions."
        ),
    )
    seed = st.sidebar.number_input(
        "Random seed",
        value=42,
        step=1,
        help="Seed for the multi-start optimizer; change for a different design.",
    )
    n_starts = st.sidebar.number_input(
        "Multi-start budget",
        min_value=1,
        value=5,
        help=(
            "Number of random initializations for each design point's "
            "optimization. Higher = more thorough search, slower."
        ),
    )

    output_dir, output_name = _output_path_picker()
    output_path = str(Path(output_dir) / output_name)

    if st.sidebar.button(
        "Generate initial design",
        type="primary",
        help=(
            "Solve the optimal-design problem, write the workbook to "
            "disk, and open it in this GUI."
        ),
    ):
        inputs = _validate_factors(factors_df)
        if inputs is None:
            return
        actual_template = _resolve_template(template_choice, len(inputs))
        if actual_template is None:
            st.sidebar.error(
                "Response surface needs exactly 2 or 3 factors; "
                f"got {len(inputs)}."
            )
            return
        out_path = Path(output_path).expanduser().resolve()
        if out_path.exists():
            st.sidebar.error(f"{out_path} already exists. Pick another path.")
            return
        params = NewParams(
            output=out_path,
            n=int(n),
            inputs=inputs,
            response_name=response_name,
            measurement_error=float(error),
            criterion=criterion,
            seed=int(seed),
            n_starts=int(n_starts),
            template=actual_template,
            degree=int(degree) if degree is not None else None,
        )
        try:
            with st.spinner("Solving optimal design..."):
                do_new(params)
        except (DoEError, ValueError) as e:
            st.sidebar.error(str(e))
            return
        _set_workbook(out_path)
        st.rerun()


def _resolve_template(gui_choice: str, n_factors: int) -> str | None:
    """Map the GUI's template menu choice to an actual `do_new` template."""
    if gui_choice == "response-surface":
        if n_factors == 2:
            return "response-surface-2d"
        if n_factors == 3:
            return "response-surface-3d"
        return None
    return gui_choice


def _output_path_picker() -> tuple[str, str]:
    """Sidebar widget: directory browser + filename → (dir, filename)."""
    st.sidebar.markdown("**Output workbook**")

    if "output_dir" not in st.session_state:
        st.session_state["output_dir"] = str(Path.cwd())

    current_dir = Path(st.session_state["output_dir"]).expanduser()
    if not current_dir.is_dir():
        current_dir = Path.cwd()
        st.session_state["output_dir"] = str(current_dir)

    dir_input = st.sidebar.text_input(
        "Folder",
        value=str(current_dir),
        key="output_dir_input",
        help=(
            "Folder where the new `.xlsx` will be written. Type a path "
            "directly, or use **Browse folder…** below to navigate. "
            "`~` is expanded."
        ),
    )
    typed = Path(dir_input).expanduser()
    if typed.is_dir() and str(typed.resolve()) != str(current_dir.resolve()):
        st.session_state["output_dir"] = str(typed.resolve())
        current_dir = typed.resolve()

    with st.sidebar.expander("Browse folder…", expanded=False):
        st.caption(f"`{current_dir}`")
        try:
            entries = sorted(current_dir.iterdir(), key=lambda p: p.name.lower())
            subdirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
            xlsx_files = [
                p for p in entries if p.is_file() and p.suffix.lower() == ".xlsx"
            ]
        except (OSError, PermissionError) as e:
            st.error(f"Cannot list folder: {e}")
            subdirs, xlsx_files = [], []

        if st.button(
            "⬆ Parent folder",
            key="browse_parent",
            disabled=current_dir.parent == current_dir,
            help="Navigate up one directory.",
        ):
            st.session_state["output_dir"] = str(current_dir.parent)
            st.rerun()

        if subdirs:
            st.caption("Subfolders")
            for d in subdirs[:50]:
                if st.button(f"📁 {d.name}", key=f"browse_dir_{d}"):
                    st.session_state["output_dir"] = str(d)
                    st.rerun()
            if len(subdirs) > 50:
                st.caption(f"…and {len(subdirs) - 50} more (type the path above).")
        else:
            st.caption("_No subfolders._")

        if xlsx_files:
            st.caption("Existing `.xlsx` files (read-only preview):")
            for f in xlsx_files[:20]:
                st.text(f"• {f.name}")

    output_name = st.sidebar.text_input(
        "Filename",
        value="campaign.xlsx",
        key="output_name_input",
        help=(
            "Name of the new workbook (must end in `.xlsx`). Must not "
            "already exist in the chosen folder."
        ),
    )

    full = Path(st.session_state["output_dir"]) / output_name
    st.sidebar.caption(f"→ `{full}`")
    return st.session_state["output_dir"], output_name


def _validate_factors(df: pd.DataFrame) -> list[tuple[str, float, float]] | None:
    rows: list[tuple[str, float, float]] = []
    for i, row in df.iterrows():
        name = str(row.get("name") or "").strip()
        lb, ub = row.get("lb"), row.get("ub")
        all_blank = not name and pd.isna(lb) and pd.isna(ub)
        if all_blank:
            continue
        if not name:
            st.sidebar.error(f"Row {int(i) + 1}: factor name is required.")
            return None
        if pd.isna(lb) or pd.isna(ub):
            st.sidebar.error(
                f"Factor {name!r}: both lower and upper bounds are required."
            )
            return None
        if float(ub) <= float(lb):
            st.sidebar.error(f"Factor {name!r}: upper bound must exceed lower bound.")
            return None
        rows.append((name, float(lb), float(ub)))
    if not rows:
        st.sidebar.error("Define at least one factor.")
        return None
    return rows


# ──────────────────────────────────────────────────────────────────
# Main pane
# ──────────────────────────────────────────────────────────────────


def _main_pane() -> None:
    path = st.session_state.get("workbook_path")
    if not path:
        _empty_state()
        return

    status = _safe_status(path)
    if status is None:
        st.error(st.session_state.get("last_error") or "Could not open workbook.")
        if st.button("Forget this workbook"):
            _set_workbook(None)
            st.rerun()
        return

    _status_banner(status)
    _runs_editor(path, status)
    _fit_panel(path, status)
    _extend_panel(path, status)
    _history_panel(path)
    _download_panel(path)


def _empty_state() -> None:
    st.title("discopt doe")
    st.markdown(
        "Pick **Open existing** to load an `.xlsx` campaign, or **New "
        "campaign** to generate a fresh optimal design."
    )
    st.markdown(
        "The CLI shortcut is `discopt doe gui PATH.xlsx` — that drops you "
        "straight into the right workbook."
    )


def _status_banner(status: dict[str, Any]) -> None:
    label = status["template"] or status["module_callable"] or "(unknown model)"
    st.subheader(f"Workbook: `{Path(status['workbook_path']).name}`")
    cols = st.columns(4)
    cols[0].metric("Model", label)
    cols[1].metric("Completed", f"{status['n_completed']} / {status['n_total']}")
    cols[2].metric("Pending", status["n_pending"])
    cols[3].metric(
        "Parameters",
        len(status["parameters"]) if status["parameters"] else 0,
    )

    inputs_str = ", ".join(
        f"{s['name']} ∈ [{s['lb']}, {s['ub']}]" for s in status["input_specs"]
    )
    st.caption(f"**Response**: `{status['response_name']}`  •  **Factors**: {inputs_str}")
    st.caption(f"**Next step (CLI)**: `{status['next_command']}`")


def _runs_editor(path: str, status: dict[str, Any]) -> None:
    st.markdown("### Runs")
    df = _runs_df(path)
    if df.empty:
        st.info("No runs yet. Generate an initial design from the sidebar.")
        return

    response = status["response_name"]
    input_cols = [s["name"] for s in status["input_specs"]]
    locked = [c for c in df.columns if c not in (response,)]
    edited = st.data_editor(
        df,
        disabled=locked,
        num_rows="fixed",
        width="stretch",
        key=f"runs_editor_{path}",
        column_config={
            response: st.column_config.NumberColumn(
                response, help=f"Measured value for '{response}' (blank = pending)"
            ),
            **{
                c: st.column_config.NumberColumn(c, disabled=True) for c in input_cols
            },
        },
    )

    col_a, col_b = st.columns([1, 1])
    if col_a.button("Save responses", type="primary"):
        n = _save_responses(path, edited, response)
        if n == 0:
            st.info("No changes detected.")
        else:
            st.success(f"Updated {n} cell(s).")
            st.rerun()
    if col_b.button("Reload from disk"):
        st.rerun()


def _fit_panel(path: str, status: dict[str, Any]) -> None:
    st.markdown("### Fit")
    can_fit = status["n_completed"] > 0 and status["template"] is not None

    if not can_fit:
        if status["module_callable"]:
            st.info(
                "Fitting for `--module` experiments is not yet supported from "
                "the GUI. Use `discopt.estimate.estimate_parameters` in Python."
            )
        else:
            st.info("No completed runs yet — fill in some response values first.")
        return

    if st.button(
        f"Fit ({status['n_completed']} observation(s))", type="primary", key="fit_btn"
    ):
        try:
            with st.spinner("Fitting parameters..."):
                do_fit({"workbook": path})
        except (DoEError, ValueError) as e:
            st.error(str(e))
            return
        st.rerun()

    if status["parameters"]:
        rows = []
        for p in status["parameters"]:
            rows.append(
                {
                    "name": p["name"],
                    "estimate": p["estimate"],
                    "std_error": p["std_error"],
                    "ci_lower_95": p["ci_lower_95"],
                    "ci_upper_95": p["ci_upper_95"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _extend_panel(path: str, status: dict[str, Any]) -> None:
    st.markdown("### Extend (next D-optimal batch)")
    if status["n_pending"] > 0:
        st.info(
            f"{status['n_pending']} run(s) still pending. Fill them in and refit "
            "before extending so the new batch can build on what you learned."
        )
        return
    n = st.number_input("Number of new runs", min_value=1, value=4, key="extend_n")
    n_starts = st.number_input(
        "Multi-start budget", min_value=1, value=5, key="extend_starts"
    )
    if st.button(f"Append {int(n)} run(s)", type="primary", key="extend_btn"):
        try:
            with st.spinner("Solving next-batch design..."):
                do_extend(
                    ExtendParams(
                        workbook=Path(path), n=int(n), n_starts=int(n_starts)
                    )
                )
        except (DoEError, ValueError) as e:
            st.error(str(e))
            return
        st.rerun()


def _history_panel(path: str) -> None:
    rows = _history_rows(path)
    if not rows:
        return
    with st.expander(f"History ({len(rows)} entr{'y' if len(rows) == 1 else 'ies'})"):
        recent = rows[-10:]
        formatted = []
        for r in recent:
            args = r.get("args")
            if isinstance(args, str):
                try:
                    args = json.dumps(json.loads(args))
                except (ValueError, TypeError):
                    pass
            formatted.append(
                {
                    "timestamp": r.get("timestamp"),
                    "command": r.get("command"),
                    "args": args,
                }
            )
        st.dataframe(
            pd.DataFrame(formatted), width="stretch", hide_index=True
        )


def _download_panel(path: str) -> None:
    p = Path(path)
    if not p.is_file():
        return
    st.download_button(
        "Download workbook",
        data=p.read_bytes(),
        file_name=p.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="discopt doe", layout="wide")
    _init_state()
    _sidebar()
    _main_pane()


if __name__ == "__main__":
    main()
