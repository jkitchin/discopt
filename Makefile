# discopt Makefile
#
# Builds the Rust extension, runs benchmarks, and saves timestamped results.
#
# Usage:
#   make benchmarks          # Full pipeline: build + lint + test + bench
#   make bench-notebook      # Just the notebook benchmark (after build)
#   make bench-smoke         # Quick smoke benchmark
#   make bench-phase3-gate   # Phase 3 gate validation
#   make bench-cutest         # Full CUTEst suite (n<=100)
#   make bench-cutest-smoke   # Quick CUTEst smoke (10 problems)
#   make setup-cutest         # Install CUTEst/SIFDecode/SIF libraries
#   make build               # Rebuild Rust .so if sources changed
#   make test                # PR-fast pytest suite (matches CI python-fast job)
#   make test-all            # Full pytest suite (slow + correctness)
#   make test-quick          # unit + smoke only (dev inner loop, target <60s)
#   make test-slow           # only the slow-marked tests
#   make test-correctness    # known-optima validation suite
#   make test-modeling       # modeling layer slice (PR-fast)
#   make test-solvers        # solver/B&B/OA slice (PR-fast)
#   make test-amp            # AMP / DOE / discrimination slice (PR-fast)
#   make test-nn             # NN embedding slice (PR-fast)
#   make test-convexity      # convexity certification slice (PR-fast)
#   make test-jax            # JAX compiler / relaxation slice (PR-fast)
#   make test-llm            # LLM modules slice (PR-fast)
#   make lint                # Ruff lint + format check
#   make clean               # Remove build artifacts
#
# Results are saved to results/ with ISO-8601 timestamps.

SHELL := /bin/bash

# --- Configuration -----------------------------------------------------------

PYTHON      ?= python
MATURIN     ?= maturin
PYTEST      ?= pytest
RUFF        ?= ruff
JUPYTER     ?= jupyter

PROJECT_DIR := $(shell pwd)
RESULTS_DIR := $(PROJECT_DIR)/results
NOTEBOOK    := docs/notebooks/benchmarks_by_class.ipynb
SO_TARGET   := python/discopt/_rust.cpython-312-darwin.so

# Timestamp for output files
TS := $(shell date -u +%Y-%m-%dT%H-%M-%S)

# Rust sources that trigger a rebuild
RUST_SRCS := $(shell find crates/ -name '*.rs' -o -name 'Cargo.toml') Cargo.toml

# JAX environment
export JAX_PLATFORMS ?= cpu
export JAX_ENABLE_X64 ?= 1

# Silence pycutest runtime compilation warnings:
# - LDFLAGS: macOS linker version mismatch (Python sysconfig bakes in 11.0)
# - FFLAGS: deprecated Fortran 77 constructs in old SIF problem files
export LDFLAGS ?= -Wl,-w
export FFLAGS  ?= -w

# CUTEst settings
CUTEST_MAX_N    ?= 100
CUTEST_PREFIX   ?= $(HOME)/.local/cutest
CUTEST_ENV      := $(CUTEST_PREFIX)/env.sh

# --- Phony targets ------------------------------------------------------------

.PHONY: all benchmarks build test test-all test-quick test-slow test-correctness \
        test-modeling test-solvers test-amp test-nn test-convexity test-jax test-llm \
        lint clean help \
        bench-notebook bench-smoke bench-phase3-gate bench-tests \
        bench-cutest bench-cutest-smoke setup-cutest check-cutest \
        docs docs-open notebooks \
        bench-lp-smoke bench-qp-smoke bench-milp-smoke bench-miqp-smoke bench-minlp-smoke bench-global-smoke \
        bench-lp-full bench-qp-full bench-milp-full bench-miqp-full bench-minlp-full bench-global-full \
        bench-smoke-all bench-full-all bench-all

all: benchmarks

help:
	@echo "discopt Makefile targets:"
	@echo ""
	@echo "  make benchmarks         Full pipeline: build, lint, test, all benchmarks"
	@echo "  make build              Rebuild Rust .so if sources changed"
	@echo "  make test               PR-fast pytest suite (matches CI python-fast)"
	@echo "  make test-all           Full pytest suite (slow + correctness)"
	@echo "  make test-quick         unit + smoke only (dev loop, target <60s)"
	@echo "  make test-slow          Only the slow-marked tests"
	@echo "  make test-correctness   Known-optima validation suite"
	@echo "  make test-modeling      Modeling layer slice"
	@echo "  make test-solvers       Solver/B&B/OA slice"
	@echo "  make test-amp           AMP / DOE / discrimination slice"
	@echo "  make test-nn            NN embedding slice"
	@echo "  make test-convexity     Convexity certification slice"
	@echo "  make test-jax           JAX compiler / relaxation slice"
	@echo "  make test-llm           LLM modules slice"
	@echo "  make lint               Ruff lint + format check"
	@echo "  make bench-notebook     Run benchmark notebook, save HTML + JSON"
	@echo "  make bench-smoke        Quick smoke benchmark via run_benchmarks.py"
	@echo "  make bench-phase3-gate  Phase 3 gate validation script"
	@echo "  make bench-tests        Run benchmark test suite"
	@echo "  make bench-cutest       Full CUTEst suite (n<=$(CUTEST_MAX_N), override with CUTEST_MAX_N=N)"
	@echo "  make bench-cutest-smoke Quick CUTEst smoke test (10 problems)"
	@echo "  make setup-cutest       Install CUTEst/SIFDecode/SIF (one-time setup)"
	@echo "  make notebooks          Execute all notebooks in place (docs/notebooks/ + manuscript/)"
	@echo "  make docs               Build Jupyter Book documentation"
	@echo "  make docs-open          Build and open Jupyter Book in browser"
	@echo "  make clean              Remove build artifacts"
	@echo ""
	@echo "Per-category benchmarks:"
	@echo "  make bench-lp-smoke     LP smoke benchmarks"
	@echo "  make bench-qp-smoke     QP smoke benchmarks"
	@echo "  make bench-milp-smoke   MILP smoke benchmarks"
	@echo "  make bench-miqp-smoke   MIQP smoke benchmarks"
	@echo "  make bench-minlp-smoke  MINLP smoke benchmarks"
	@echo "  make bench-global-smoke Global opt smoke benchmarks"
	@echo "  make bench-lp-full      LP full benchmarks"
	@echo "  make bench-qp-full      QP full benchmarks"
	@echo "  make bench-milp-full    MILP full benchmarks"
	@echo "  make bench-miqp-full    MIQP full benchmarks"
	@echo "  make bench-minlp-full   MINLP full benchmarks"
	@echo "  make bench-global-full  Global opt full benchmarks"
	@echo "  make bench-smoke-all    All smoke benchmarks"
	@echo "  make bench-full-all     All full benchmarks"
	@echo "  make bench-all          All benchmarks (smoke + full)"
	@echo ""
	@echo "Results are saved to results/ with timestamps."
	@echo ""
	@echo "CUTEst:"
	@echo "  Run 'make setup-cutest' once, then 'source $(CUTEST_ENV)'"
	@echo "  before running bench-cutest targets."

# --- Build --------------------------------------------------------------------

# Rebuild the Rust extension only when sources are newer than the .so
$(SO_TARGET): $(RUST_SRCS)
	@echo "==> Rebuilding Rust extension (sources changed)..."
	$(MATURIN) develop --release
	@# maturin develop may install to site-packages; copy to project dir
	@SP=$$($(PYTHON) -c "import sysconfig; print(sysconfig.get_path('purelib'))"); \
	if [ -f "$$SP/discopt/_rust.cpython-312-darwin.so" ]; then \
		cp "$$SP/discopt/_rust.cpython-312-darwin.so" $(SO_TARGET); \
		echo "==> Copied .so from site-packages"; \
	fi
	@touch $(SO_TARGET)
	@echo "==> Rust extension ready"

build: $(SO_TARGET)

# --- Lint ---------------------------------------------------------------------

lint:
	@echo "==> Running ruff lint..."
	$(RUFF) check python/
	@echo "==> Running ruff format check..."
	$(RUFF) format --check python/
	@echo "==> Lint passed"

# --- Test ---------------------------------------------------------------------
#
# Tiers (issue #68):
#   test          — PR-fast: matches CI python-fast job. Excludes `slow` and the
#                   correctness suite. Target <5 min.
#   test-all      — everything, including slow + correctness. Target ~20 min.
#   test-quick    — unit + smoke only, dev inner loop. Target <60 s.
#   test-slow     — only slow-marked tests (full backend cross-product etc.).
#   test-correctness — full known-optima validation suite (test_correctness.py).
#   test-<slice>  — subject-area slice run with the PR-fast filter applied.

# Common flags for the PR-fast tier (kept in sync with .github/workflows/ci.yml).
# `not slow` excludes the heavy backend matrix; the curated PR correctness
# subset (test_pr_correctness.py) is *not* slow-marked so it still runs.
PYTEST_FAST_FLAGS := --timeout=120 -m "not slow" \
    --ignore=python/tests/test_correctness.py

# File groups for slice targets. A test file may appear in more than one group.
TEST_MODELING := \
    python/tests/test_export.py \
    python/tests/test_dag_compiler.py \
    python/tests/test_rust_ir.py \
    python/tests/test_fast_construction.py \
    python/tests/test_gams.py \
    python/tests/test_gdp.py \
    python/tests/test_model_selection.py \
    python/tests/test_nl_parser.py \
    python/tests/test_nl_reconstruction.py \
    python/tests/test_nl_writer.py

TEST_SOLVERS := \
    python/tests/test_alphabb.py \
    python/tests/test_cutting_planes.py \
    python/tests/test_fbbt_bindings.py \
    python/tests/test_gdpopt_loa.py \
    python/tests/test_ipm.py \
    python/tests/test_ipm_callbacks.py \
    python/tests/test_ipm_iterative.py \
    python/tests/test_lp_highs.py \
    python/tests/test_lp_qp_solvers.py \
    python/tests/test_minlplib_benchmark.py \
    python/tests/test_minlptests.py \
    python/tests/test_nlp_bb.py \
    python/tests/test_nlp_convergence.py \
    python/tests/test_nlp_evaluator.py \
    python/tests/test_nlp_ipopt.py \
    python/tests/test_oa.py \
    python/tests/test_obbt.py \
    python/tests/test_orchestrator.py \
    python/tests/test_primal_heuristics.py \
    python/tests/test_qp_highs.py \
    python/tests/test_sparse_ipm.py \
    python/tests/test_t24_batch_ipm.py \
    python/tests/test_tree.py \
    python/tests/test_warm_start.py

TEST_AMP := \
    python/tests/test_affine_decision_rule.py \
    python/tests/test_amp.py \
    python/tests/test_batch_dispatch.py \
    python/tests/test_batch_doe.py \
    python/tests/test_batch_evaluator.py \
    python/tests/test_discrimination_criteria.py \
    python/tests/test_discrimination_examples.py \
    python/tests/test_discrimination_sequential.py \
    python/tests/test_doe.py \
    python/tests/test_estimability.py \
    python/tests/test_estimate.py \
    python/tests/test_fim.py \
    python/tests/test_identifiability.py \
    python/tests/test_identifiability_edge_cases.py \
    python/tests/test_robust_counterpart.py \
    python/tests/test_robust_solve.py \
    python/tests/test_robust_uncertainty.py \
    python/tests/test_sequential_doe.py

TEST_NN := \
    python/tests/test_gnn_branching.py \
    python/tests/test_learned_relaxations.py \
    python/tests/test_nn_formulations.py

TEST_CONVEXITY := \
    python/tests/test_convex_fast_path.py \
    python/tests/test_convexity.py \
    python/tests/test_convexity_certificate.py \
    python/tests/test_convexity_eigenvalue.py \
    python/tests/test_convexity_interval.py \
    python/tests/test_convexity_interval_ad.py \
    python/tests/test_convexity_interval_eval.py \
    python/tests/test_convexity_lattice.py \
    python/tests/test_convexity_node_refresh.py \
    python/tests/test_convexity_pathological.py \
    python/tests/test_convexity_solver_integration.py \
    python/tests/test_convexity_soundness.py \
    python/tests/test_convexity_wide_box.py

TEST_JAX := \
    python/tests/test_dag_compiler.py \
    python/tests/test_differentiable.py \
    python/tests/test_envelopes.py \
    python/tests/test_mccormick.py \
    python/tests/test_mccormick_bounds.py \
    python/tests/test_piecewise_mccormick.py \
    python/tests/test_relaxation_compiler.py \
    python/tests/test_sparse_coo.py \
    python/tests/test_sparsity.py \
    python/tests/test_trilinear_exact.py

TEST_LLM := \
    python/tests/test_llm_modules.py

# PR-fast: matches python-fast CI job. This is what `make test` should mean.
test: build
	@echo "==> Running PR-fast pytest suite (matches CI python-fast)..."
	$(PYTEST) python/tests/ -v --tb=short -q $(PYTEST_FAST_FLAGS)
	@echo "==> PR-fast tests passed"

# Full suite: every test, no exclusions. Use before releases or when triaging.
test-all: build
	@echo "==> Running full pytest suite (slow + correctness + everything)..."
	$(PYTEST) python/tests/ -v --tb=short -q
	@echo "==> Full suite passed"

# Dev inner loop: only unit and smoke markers. Wired by Phase 3 of issue #68;
# may be near-empty until those markers are populated.
test-quick: build
	@echo "==> Running quick tests (unit + smoke)..."
	$(PYTEST) python/tests/ -v --tb=short -q --timeout=60 -m "unit or smoke"
	@echo "==> Quick tests passed"

# Only the slow-marked tests (backend cross-product, big instances, ML training).
test-slow: build
	@echo "==> Running slow-marked tests..."
	$(PYTEST) python/tests/ -v --tb=short -q -m "slow"
	@echo "==> Slow tests passed"

# Full known-optima validation. Heavy; not in PR gate.
test-correctness: build
	@echo "==> Running correctness suite (known-optima validation)..."
	$(PYTEST) python/tests/test_correctness.py -v --tb=short -q
	@echo "==> Correctness suite passed"

# Slice targets: PR-fast filter applied within a subject area.
test-modeling: build
	$(PYTEST) $(TEST_MODELING) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-solvers: build
	$(PYTEST) $(TEST_SOLVERS) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-amp: build
	$(PYTEST) $(TEST_AMP) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-nn: build
	$(PYTEST) $(TEST_NN) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-convexity: build
	$(PYTEST) $(TEST_CONVEXITY) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-jax: build
	$(PYTEST) $(TEST_JAX) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-llm: build
	$(PYTEST) $(TEST_LLM) -v --tb=short -q $(PYTEST_FAST_FLAGS)

# --- Results directory --------------------------------------------------------

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# --- Benchmark: Notebook ------------------------------------------------------

bench-notebook: build | $(RESULTS_DIR)
	@echo "==> Running benchmark notebook..."
	$(JUPYTER) nbconvert \
		--to notebook --execute \
		--ExecutePreprocessor.timeout=600 \
		--output-dir=$(RESULTS_DIR) \
		--output=benchmarks_$(TS).ipynb \
		$(NOTEBOOK)
	@echo "==> Converting to HTML..."
	$(JUPYTER) nbconvert \
		--to html \
		$(RESULTS_DIR)/benchmarks_$(TS).ipynb \
		--output benchmarks_$(TS).html
	@echo "==> Extracting benchmark data..."
	$(PYTHON) scripts/extract_notebook_results.py \
		$(RESULTS_DIR)/benchmarks_$(TS).ipynb \
		$(RESULTS_DIR)/benchmarks_$(TS).json
	@echo "==> Notebook benchmark complete: $(RESULTS_DIR)/benchmarks_$(TS).*"

# --- Benchmark: Smoke ---------------------------------------------------------

bench-smoke: build | $(RESULTS_DIR)
	@echo "==> Running smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite smoke \
		--output $(RESULTS_DIR)/smoke_$(TS).json
	@echo "==> Smoke benchmark saved to $(RESULTS_DIR)/smoke_$(TS).json"

# --- Benchmark: Phase 3 gate -------------------------------------------------

bench-phase3-gate: build | $(RESULTS_DIR)
	@echo "==> Running Phase 3 gate validation..."
	$(PYTHON) scripts/phase3_gate.py \
		--time-limit 60 --max-nodes 100000 2>&1 \
		| tee $(RESULTS_DIR)/phase3_gate_$(TS).log
	@# The script saves its own JSON to reports/; copy it
	@LATEST=$$(ls -t reports/phase3_gate_*.json 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		cp "$$LATEST" $(RESULTS_DIR)/phase3_gate_$(TS).json; \
		echo "==> Phase 3 gate results: $(RESULTS_DIR)/phase3_gate_$(TS).json"; \
	fi

# --- Benchmark: Test suite (pytest benchmarks) --------------------------------

bench-tests: build | $(RESULTS_DIR)
	@echo "==> Running benchmark test suite..."
	$(PYTEST) discopt_benchmarks/tests/ -v --tb=short -q \
		--junitxml=$(RESULTS_DIR)/bench_tests_$(TS).xml 2>&1 \
		| tee $(RESULTS_DIR)/bench_tests_$(TS).log
	@echo "==> Benchmark tests saved to $(RESULTS_DIR)/bench_tests_$(TS).*"

# --- CUTEst setup -------------------------------------------------------------

$(CUTEST_ENV):
	@echo "==> CUTEst not installed at $(CUTEST_PREFIX)"
	@echo "    Run 'make setup-cutest' first."
	@exit 1

setup-cutest:
	@echo "==> Installing CUTEst libraries to $(CUTEST_PREFIX)..."
	CUTEST_PREFIX=$(CUTEST_PREFIX) bash scripts/setup_cutest.sh
	@echo ""
	@echo "  Now add to your shell profile (e.g. ~/.zshrc):"
	@echo "    source $(CUTEST_ENV)"
	@echo ""
	@echo "  Then reopen your terminal or run:"
	@echo "    source $(CUTEST_ENV)"

# Check that CUTEst env is active (used as a prerequisite)
check-cutest:
	@$(PYTHON) -c "import pycutest" 2>/dev/null || { \
		echo ""; \
		echo "ERROR: pycutest cannot find CUTEst libraries."; \
		echo ""; \
		if [ -f "$(CUTEST_ENV)" ]; then \
			echo "  CUTEst is installed but env vars are not set."; \
			echo "  Run:  source $(CUTEST_ENV)"; \
		else \
			echo "  CUTEst is not installed."; \
			echo "  Run:  make setup-cutest"; \
			echo "  Then: source $(CUTEST_ENV)"; \
		fi; \
		echo ""; \
		exit 1; \
	}
	@echo "==> CUTEst environment OK"

# --- Benchmark: CUTEst (full) -------------------------------------------------

bench-cutest: build check-cutest | $(RESULTS_DIR)
	@echo "==> Running full CUTEst benchmark (n <= $(CUTEST_MAX_N))..."
	$(PYTHON) scripts/run_cutest_comprehensive.py \
		--max-n $(CUTEST_MAX_N) \
		--output $(RESULTS_DIR)/cutest_$(TS).json 2>&1 \
		| tee $(RESULTS_DIR)/cutest_$(TS).log
	@echo "==> Generating CUTEst report notebook..."
	$(PYTHON) scripts/generate_cutest_report.py \
		$(RESULTS_DIR)/cutest_$(TS).json \
		$(RESULTS_DIR)/cutest_$(TS).ipynb
	@echo "==> CUTEst report: $(RESULTS_DIR)/cutest_$(TS).ipynb"
	@echo "==> CUTEst results: $(RESULTS_DIR)/cutest_$(TS).{json,log,ipynb}"

# --- Benchmark: CUTEst (smoke) -----------------------------------------------

bench-cutest-smoke: build check-cutest | $(RESULTS_DIR)
	@echo "==> Running CUTEst smoke test..."
	$(PYTHON) scripts/run_cutest_comprehensive.py \
		--smoke \
		--output $(RESULTS_DIR)/cutest_smoke_$(TS).json 2>&1 \
		| tee $(RESULTS_DIR)/cutest_smoke_$(TS).log
	@echo "==> Generating CUTEst report notebook..."
	$(PYTHON) scripts/generate_cutest_report.py \
		$(RESULTS_DIR)/cutest_smoke_$(TS).json \
		$(RESULTS_DIR)/cutest_smoke_$(TS).ipynb
	@echo "==> CUTEst smoke report: $(RESULTS_DIR)/cutest_smoke_$(TS).ipynb"
	@echo "==> CUTEst smoke results: $(RESULTS_DIR)/cutest_smoke_$(TS).{json,log,ipynb}"

# --- Full pipeline ------------------------------------------------------------

benchmarks: build lint test-all bench-notebook bench-smoke
	@echo ""
	@echo "============================================================"
	@echo "  All benchmarks complete.  Results in: $(RESULTS_DIR)/"
	@echo "  Timestamp: $(TS)"
	@echo "============================================================"
	@ls -lh $(RESULTS_DIR)/*$(TS)* 2>/dev/null

# --- Notebooks (run in place) ------------------------------------------------

# Source notebooks (docs + manuscript, excluding build artifacts)
NB_SOURCES := $(wildcard docs/notebooks/*.ipynb) $(wildcard manuscript/*.ipynb)

notebooks: build
	@echo "==> Running $(words $(NB_SOURCES)) notebooks in place..."
	@failed=0; \
	for nb in $(NB_SOURCES); do \
		echo "  -> $$nb"; \
		$(JUPYTER) nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=600 \
			"$$nb" || { echo "  !! FAILED: $$nb"; failed=$$((failed + 1)); }; \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo "==> $$failed notebook(s) failed"; exit 1; \
	fi
	@echo "==> All notebooks executed successfully"

# --- Documentation -----------------------------------------------------------

docs:
	@echo "==> Building Jupyter Book..."
	jupyter-book build docs/
	@echo "==> Jupyter Book built: docs/_build/html/index.html"

docs-open: docs
	@echo "==> Opening Jupyter Book in browser..."
	open docs/_build/html/index.html

# --- Per-Category Benchmarks --------------------------------------------------

bench-lp-smoke: build | $(RESULTS_DIR)
	@echo "==> Running LP smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category lp --level smoke --report --html \
		--output $(RESULTS_DIR)/lp_smoke_$(TS)
	@echo "==> LP smoke benchmark complete"

bench-qp-smoke: build | $(RESULTS_DIR)
	@echo "==> Running QP smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category qp --level smoke --report --html \
		--output $(RESULTS_DIR)/qp_smoke_$(TS)

bench-milp-smoke: build | $(RESULTS_DIR)
	@echo "==> Running MILP smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category milp --level smoke --report --html \
		--output $(RESULTS_DIR)/milp_smoke_$(TS)

bench-miqp-smoke: build | $(RESULTS_DIR)
	@echo "==> Running MIQP smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category miqp --level smoke --report --html \
		--output $(RESULTS_DIR)/miqp_smoke_$(TS)

bench-minlp-smoke: build | $(RESULTS_DIR)
	@echo "==> Running MINLP smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category minlp --level smoke --report --html \
		--output $(RESULTS_DIR)/minlp_smoke_$(TS)

bench-global-smoke: build | $(RESULTS_DIR)
	@echo "==> Running global opt smoke benchmarks..."
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category global_opt --level smoke --report --html \
		--output $(RESULTS_DIR)/global_smoke_$(TS)

bench-lp-full: build | $(RESULTS_DIR)
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category lp --level full --report --html \
		--output $(RESULTS_DIR)/lp_full_$(TS)

bench-qp-full: build | $(RESULTS_DIR)
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category qp --level full --report --html \
		--output $(RESULTS_DIR)/qp_full_$(TS)

bench-milp-full: build | $(RESULTS_DIR)
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category milp --level full --report --html \
		--output $(RESULTS_DIR)/milp_full_$(TS)

bench-miqp-full: build | $(RESULTS_DIR)
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category miqp --level full --report --html \
		--output $(RESULTS_DIR)/miqp_full_$(TS)

bench-minlp-full: build | $(RESULTS_DIR)
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category minlp --level full --report --html \
		--output $(RESULTS_DIR)/minlp_full_$(TS)

bench-global-full: build | $(RESULTS_DIR)
	$(PYTHON) discopt_benchmarks/run_category_benchmarks.py \
		--category global_opt --level full --report --html \
		--output $(RESULTS_DIR)/global_full_$(TS)

bench-smoke-all: bench-lp-smoke bench-qp-smoke bench-milp-smoke bench-miqp-smoke bench-minlp-smoke bench-global-smoke
	@echo "==> All smoke benchmarks complete"

bench-full-all: bench-lp-full bench-qp-full bench-milp-full bench-miqp-full bench-minlp-full bench-global-full
	@echo "==> All full benchmarks complete"

bench-all: bench-smoke-all bench-full-all
	@echo "==> All benchmarks complete"

# --- Clean --------------------------------------------------------------------

clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf target/debug target/release
	rm -f $(SO_TARGET)
	@echo "==> Clean complete"
