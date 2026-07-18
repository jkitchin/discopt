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
#   make gams-install        # Build + register discopt as a GAMS solver
#   make gams-build          # Install discopt with the GAMS link (gamsapi[core])
#   make gams-register       # Register discopt in GAMS_CONFIG_DIR (default ~/.gams)
#   make gams-test           # Run the GAMS link test suite
#   make gams-verify         # End-to-end GAMS link smoke test (needs a GAMS install)
#   make test                # PR-fast pytest suite (matches CI python-fast job)
#   make test-all            # Full pytest suite (slow + correctness)
#   make test-quick          # unit + smoke only (dev inner loop, target <60s)
#   make test-slow           # only the slow-marked tests
#   make test-correctness    # known-optima validation suite
#   make test-modeling       # modeling layer slice (PR-fast)
#   make test-solvers        # solver/B&B/OA slice (PR-fast)
#   make test-amp            # AMP slice (PR-fast)
#   make test-amp-fast       # Run fast AMP regression battery
#   make test-amp-integration # Run opt-in AMP Alpine/incidence suite
#   make test-nn             # NN embedding slice (PR-fast)
#   make test-convexity      # convexity certification slice (PR-fast)
#   make test-jax            # JAX compiler / relaxation slice (PR-fast)
#   make test-llm            # LLM modules slice (PR-fast)
#   make lint                # Ruff lint + format check
#   make hooks               # Install pre-commit hooks
#   make clean               # Remove build artifacts
#
# Results are saved to results/ with ISO-8601 timestamps.

SHELL := /bin/bash

# --- Configuration -----------------------------------------------------------

PYTHON      ?= python
MATURIN     ?= maturin
PYTEST      ?= pytest
PYTEST_MEMORY_CAP ?= scripts/run_memory_capped_pytest.sh
PYTEST_MEMORY_LIMIT_MB ?= 32768
PYTEST_CPU_LIMIT_SECONDS ?= 0
PYTEST_XDIST_WORKERS ?=
PYTEST_CAPPED = PYTEST_MEMORY_LIMIT_MB=$(PYTEST_MEMORY_LIMIT_MB) PYTEST_CPU_LIMIT_SECONDS=$(PYTEST_CPU_LIMIT_SECONDS) $(PYTEST_MEMORY_CAP) $(PYTEST)
RUFF        ?= ruff
JUPYTER     ?= jupyter
PRE_COMMIT  ?= pre-commit

PROJECT_DIR := $(shell pwd)
RESULTS_DIR := $(PROJECT_DIR)/results
NOTEBOOK    := docs/notebooks/benchmarks_by_class.ipynb
EXT_SUFFIX  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
SO_TARGET   := python/discopt/_rust$(EXT_SUFFIX)

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

.PHONY: all benchmarks build test test-fast test-all test-quick test-slow test-correctness \
        test-modeling test-solvers test-amp test-amp-fast test-amp-integration \
        test-nn test-convexity test-jax test-llm \
        perf-gate perf-baseline \
        lint hooks clean help \
        bench-notebook bench-smoke bench-phase3-gate bench-tests \
        bench-cutest bench-cutest-smoke setup-cutest check-cutest \
        docs docs-open notebooks \
        gams-build gams-register gams-install gams-test gams-verify \
        graduation-gate graduation-gate-ci \
        bench-lp-smoke bench-qp-smoke bench-milp-smoke bench-miqp-smoke bench-minlp-smoke bench-global-smoke \
        bench-lp-full bench-qp-full bench-milp-full bench-miqp-full bench-minlp-full bench-global-full \
        bench-smoke-all bench-full-all bench-all bench-global-baron bench-global-nlsolvers

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
	@echo "  make perf-gate          Perf regression gate: panel vs baseline (~5 min)"
	@echo "  make perf-baseline      Regenerate the committed perf baseline"
	@echo "  make test-correctness   Known-optima validation suite"
	@echo "  make test-modeling      Modeling layer slice"
	@echo "  make test-solvers       Solver/B&B/OA slice"
	@echo "  make test-amp           AMP slice"
	@echo "  make test-amp-fast      Run fast AMP regression battery"
	@echo "  make test-amp-integration Run opt-in AMP Alpine/incidence suite"
	@echo "  make test-nn            NN embedding slice"
	@echo "  make test-convexity     Convexity certification slice"
	@echo "  make test-jax           JAX compiler / relaxation slice"
	@echo "  make test-llm           LLM modules slice"
	@echo "  make lint               Ruff lint + format check"
	@echo "  make hooks              Install pre-commit hooks"
	@echo "  make gams-install       Build + register discopt as a GAMS solver"
	@echo "  make gams-build         Install discopt with the GAMS link (gamsapi[core])"
	@echo "  make gams-register      Register discopt in GAMS_CONFIG_DIR (default ~/.gams)"
	@echo "  make gams-test          Run the GAMS link test suite"
	@echo "  make gams-verify        End-to-end GAMS link smoke test (needs GAMS)"
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
	if [ -f "$$SP/discopt/_rust$(EXT_SUFFIX)" ]; then \
		cp "$$SP/discopt/_rust$(EXT_SUFFIX)" $(SO_TARGET); \
		echo "==> Copied .so from site-packages"; \
	fi
	@touch $(SO_TARGET)
	@echo "==> Rust extension ready"

build: $(SO_TARGET)

# --- GAMS solver link ---------------------------------------------------------
#
# Build and install discopt as a GAMS solver. `gams-install` does both:
# installs discopt with the GMO/GEV Python bindings, then registers the solver
# with your GAMS system so `option minlp = discopt;` dispatches to discopt.
#
#   make gams-install                       # build + register (default dir)
#   make gams-install GAMS_CONFIG_DIR=/path # register into a specific dir
#
# GAMS_CONFIG_DIR defaults to $(HOME)/.gams, which GAMS reads automatically.

PIP             ?= pip
GAMS_CONFIG_DIR ?= $(HOME)/.gams

gams-build:
	@echo "==> Installing discopt with the GAMS link (gamsapi[core])..."
	$(PIP) install -e ".[gams]"
	@echo "==> discopt[gams] installed"

gams-register:
	@echo "==> Registering discopt as a GAMS solver in $(GAMS_CONFIG_DIR)..."
	$(PYTHON) -m discopt.cli gams-register --directory "$(GAMS_CONFIG_DIR)"

gams-install: gams-build gams-register
	@echo "==> discopt GAMS link built and registered."
	@echo "    Merge $(GAMS_CONFIG_DIR)/gamsconfig.yaml into your GAMS config if needed,"
	@echo "    then in GAMS: option minlp = discopt; solve m using minlp minimizing z;"

gams-test:
	@echo "==> Running GAMS link test suite..."
	$(PYTEST) python/tests/test_gams_link.py -v

gams-verify:
	@echo "==> Verifying the GAMS link end-to-end (requires a GAMS install + registration)..."
	$(PYTHON) -m discopt.gams.verify

# --- Lint ---------------------------------------------------------------------

lint:
	@echo "==> Running ruff lint..."
	$(RUFF) check python/
	@echo "==> Running ruff format check..."
	$(RUFF) format --check python/
	@echo "==> Lint passed"

hooks:
	@echo "==> Installing pre-commit hooks..."
	$(PRE_COMMIT) install
	@echo "==> Hooks installed"

# --- Test ---------------------------------------------------------------------
#
# Tiers (issue #68):
#   test          - PR-fast: matches CI python-fast job. Excludes `slow`,
#                   `correctness`, `integration`, `amp_benchmark`, and
#                   `requires_cyipopt` while
#                   keeping the curated `pr_correctness` subset. Target <10 min.
#   test-all      - everything, including slow + correctness. Target ~20 min.
#   test-quick    - unit + smoke only, dev inner loop. Target <60 s.
#   test-slow     - only slow-marked tests (full backend cross-product etc.).
#   test-correctness - full known-optima validation suite (test_correctness.py).
#   test-<slice>  - subject-area slice run with the PR-fast filter applied.

# Common flags for the PR-fast tier (kept in sync with .github/workflows/ci.yml).
# These exclusions keep the PR gate focused on ordinary feature tests plus the
# curated `pr_correctness` subset. Full correctness, integration, and benchmark
# coverage stay available through the explicit targets below.
# `--dist loadgroup` keeps tests sharing a class on the same worker so
# xdist-incompatible fixtures stay serialized.
PYTEST_FAST_FLAGS := --timeout=120 -m "not slow and not correctness and not integration and not amp_benchmark and not requires_cyipopt and not memory_heavy" \
    --ignore=python/tests/test_correctness.py \
    -n $(or $(PYTEST_XDIST_WORKERS),auto) --dist loadgroup

PYTEST_QUICK_FLAGS := --timeout=60 -m "(unit or smoke) and not slow and not integration and not amp_benchmark and not requires_cyipopt and not memory_heavy"

PYTEST_AMP_FAST_FLAGS := --timeout=120 -m "not slow and not integration and not amp_benchmark and not requires_cyipopt and not memory_heavy"

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
    python/tests/test_qp_solve.py \
    python/tests/test_sparse_ipm.py \
    python/tests/test_t24_batch_ipm.py \
    python/tests/test_tree.py \
    python/tests/test_warm_start.py

TEST_AMP := \
    python/tests/test_affine_decision_rule.py \
    python/tests/test_amp.py \
    python/tests/test_batch_dispatch.py \
    python/tests/test_batch_evaluator.py \
    python/tests/test_estimate.py \
    python/tests/test_robust_counterpart.py \
    python/tests/test_robust_solve.py \
    python/tests/test_robust_uncertainty.py

TEST_NN := \
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
	$(PYTEST_CAPPED) python/tests/ -v --tb=short -q $(PYTEST_FAST_FLAGS)
	@echo "==> PR-fast tests passed"

test-fast: test

# Full suite: every test, no exclusions. Use before releases or when triaging.
test-all: build
	@echo "==> Running full pytest suite (slow + correctness + everything)..."
	$(PYTEST_CAPPED) python/tests/ -v --tb=short -q
	@echo "==> Full suite passed"

# Dev inner loop: only unit and smoke markers. Wired by Phase 3 of issue #68;
# may be near-empty until those markers are populated.
test-quick: build
	@echo "==> Running quick tests (unit + smoke)..."
	$(PYTEST) python/tests/ -v --tb=short -q $(PYTEST_QUICK_FLAGS)
	@echo "==> Quick tests passed"

# Only the slow-marked tests (backend cross-product, big instances, ML training).
test-slow: build
	@echo "==> Running slow-marked tests..."
	$(PYTEST_CAPPED) python/tests/ -v --tb=short -q -m "slow"
	@echo "==> Slow tests passed"

# Performance regression gate (perf plan Stage 0). Runs the fixed vendored panel
# and fails on any correctness violation or deterministic perf regression (>15%
# node_count / compiles-per-node) vs docs/dev/data/perf-baseline.jsonl. Nightly /
# pre-merge, NOT on the PR-fast path (~4-5 min). See docs/dev/performance-plan.md.
perf-gate: build
	@echo "==> Running performance gate (panel vs committed baseline)..."
	$(PYTHON) -m discopt_benchmarks.perf.gate
	@echo "==> Perf gate passed"

# Regenerate the committed baseline (run after an intended perf change; review the
# docs/dev/data/perf-baseline.jsonl diff before committing).
perf-baseline: build
	@echo "==> Regenerating perf baseline..."
	$(PYTHON) -m discopt_benchmarks.perf.gate --update-baseline

# Full known-optima validation. Heavy; not in PR gate.
test-correctness: build
	@echo "==> Running correctness suite (known-optima validation)..."
	$(PYTEST_CAPPED) python/tests/test_correctness.py -v --tb=short -q
	@echo "==> Correctness suite passed"

# Slice targets: PR-fast filter applied within a subject area.
test-modeling: build
	$(PYTEST_CAPPED) $(TEST_MODELING) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-solvers: build
	$(PYTEST_CAPPED) $(TEST_SOLVERS) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-amp: build
	$(PYTEST_CAPPED) $(TEST_AMP) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-amp-fast: build
	@echo "==> Running fast AMP regression tests..."
	$(PYTEST_CAPPED) python/tests/test_amp.py -v --tb=short -q $(PYTEST_AMP_FAST_FLAGS)
	@echo "==> Fast AMP tests passed"

test-amp-integration: build
	@echo "==> Running opt-in AMP Alpine/incidence tests..."
	$(PYTEST_CAPPED) python/tests/test_amp_integration.py -v --tb=short -q -m "slow or integration or amp_benchmark or requires_cyipopt or memory_heavy"
	@echo "==> AMP integration tests passed"

test-nn: build
	$(PYTEST_CAPPED) $(TEST_NN) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-convexity: build
	$(PYTEST_CAPPED) $(TEST_CONVEXITY) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-jax: build
	$(PYTEST_CAPPED) $(TEST_JAX) -v --tb=short -q $(PYTEST_FAST_FLAGS)

test-llm: build
	$(PYTEST_CAPPED) $(TEST_LLM) -v --tb=short -q $(PYTEST_FAST_FLAGS)

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
	$(PYTEST_CAPPED) discopt_benchmarks/tests/ -v --tb=short -q \
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

# Global-opt head-to-head vs full-license BARON (GAMS). Unlike the *-compare
# targets (which drive the demo-limited AMPL-ASL BARON off the .nl), this fetches
# minlplib .gms and runs `gams ... minlp=baron`, the only path to the CMU-license
# BARON. Covers all 62 vendored .nl; correctness vs MINLPLib primalbound.
# Override the budget with GLOBAL_BARON_TL (seconds, default 60).
GLOBAL_BARON_TL ?= 60

bench-global-baron: build
	@echo "==> Global-opt head-to-head: discopt vs BARON (GAMS), 62 instances, $(GLOBAL_BARON_TL)s each"
	$(PYTHON) -m discopt_benchmarks.scripts.global_opt_baron_vs_discopt \
		--time-limit $(GLOBAL_BARON_TL) --out-dir $(BENCH_OUT_DIR)

# Global-opt head-to-head vs the .nl-native open-source solvers: HiGHS, SCIP,
# Couenne. All read the vendored .nl directly (no GAMS), reusing the runner's
# command-builder + per-solver parsers. Correctness vs MINLPLib primalbound over
# all 62 vendored instances. (HiGHS is LP/MILP-only and errors on nonlinear
# instances by design — included as a reference for the linear subset.)
# Override the budget with GLOBAL_NL_TL, the solver set with GLOBAL_NL_SOLVERS.
GLOBAL_NL_TL ?= 60
GLOBAL_NL_SOLVERS ?= highs,scip,couenne

bench-global-nlsolvers: build
	@echo "==> Global-opt head-to-head: discopt vs $(GLOBAL_NL_SOLVERS), 62 instances, $(GLOBAL_NL_TL)s each"
	$(PYTHON) -m discopt_benchmarks.scripts.global_opt_nl_solvers \
		--time-limit $(GLOBAL_NL_TL) --solvers $(GLOBAL_NL_SOLVERS) --out-dir $(BENCH_OUT_DIR)

bench-smoke-all: bench-lp-smoke bench-qp-smoke bench-milp-smoke bench-miqp-smoke bench-minlp-smoke bench-global-smoke
	@echo "==> All smoke benchmarks complete"

bench-full-all: bench-lp-full bench-qp-full bench-milp-full bench-miqp-full bench-minlp-full bench-global-full
	@echo "==> All full benchmarks complete"

bench-all: bench-smoke-all bench-full-all
	@echo "==> All benchmarks complete"

# --- Stratified MINLPLib tiers (small / medium / full) ------------------------
# Three tiered suites that span every (class × size) bucket of MINLPLib.
# See discopt_benchmarks/config/suites/{small,medium,full}.txt for the lists.
#
# Per-instance time limits: 60s (small) / 300s (medium) / 600s (full)
# Target wall-clock on 8 workers: ~30 min / ~2 h / 12-30 h.
#
# Two variants per tier:
#   bench-<tier>          discopt only via the scaled (parallel, resumable) path
#   bench-<tier>-compare  multi-solver head-to-head via the in-process path

BENCH_WORKERS ?= 8
BENCH_OUT_DIR ?= reports
SOLVERS_COMPARE ?= discopt,scip,bonmin

# ── Data + solver setup (one-time) ───────────────────────────────────────────

fetch-minlplib:
	@echo "==> Fetching MINLPLib into ~/.cache/discopt/minlplib/current/"
	$(PYTHON) -m discopt_benchmarks.scripts.fetch_minlplib

fetch-minlplib-small:
	@echo "==> Fetching only the small-tier instances"
	@INST=$$(grep -v '^#' discopt_benchmarks/config/suites/small.txt | grep -v '^$$' | paste -sd, -); \
	$(PYTHON) -m discopt_benchmarks.scripts.fetch_minlplib --instances "$$INST"

make-suites:
	@echo "==> Regenerating stratified instance lists"
	$(PYTHON) -m discopt_benchmarks.scripts.make_suites --force

setup-solvers:
	@echo "==> Installing SCIP via Homebrew + Bonmin via conda-forge"
	@brew list scip >/dev/null 2>&1 || brew install scip
	@brew list --cask miniforge >/dev/null 2>&1 || brew install --cask miniforge
	@source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh; \
		conda env list | grep -q '^discopt-bench ' \
		|| mamba create -n discopt-bench -c conda-forge -y coin-or-bonmin
	@echo "==> Done. Run 'make check-solvers' to verify."

check-solvers:
	@echo "==> Checking installed solvers..."
	@command -v scip >/dev/null && scip --version | head -1 || echo "  scip:    MISSING (brew install scip)"
	@BONMIN=/opt/homebrew/Caskroom/miniforge/base/envs/discopt-bench/bin/bonmin; \
		test -x $$BONMIN && $$BONMIN -v 2>/dev/null | head -1 \
		|| echo "  bonmin:  MISSING (make setup-solvers)"
	@command -v couenne >/dev/null && couenne -v 2>/dev/null | head -1 \
		|| echo "  couenne: not installed (no conda-forge package for osx-arm64)"

# ── Small tier (~60 instances, ~30 min on 8 workers) ─────────────────────────

bench-small: build
	@echo "==> Running small tier (discopt, scaled-runner, $(BENCH_WORKERS) workers)"
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite small --use-cache --subprocess \
		--workers $(BENCH_WORKERS) --mem-limit-mb 0 \
		--scaled-out-dir $(BENCH_OUT_DIR)/small_$(TS)

bench-small-compare: build
	@echo "==> Running small tier head-to-head: $(SOLVERS_COMPARE)"
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite small --use-cache \
		--solvers $(SOLVERS_COMPARE) \
		--scaled-out-dir $(BENCH_OUT_DIR)/small_compare_$(TS)

# ── Medium tier (~250 instances, ~2 h on 8 workers) ──────────────────────────

bench-medium: build
	@echo "==> Running medium tier (discopt, scaled-runner)"
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite medium --use-cache --subprocess \
		--workers $(BENCH_WORKERS) --mem-limit-mb 0 \
		--scaled-out-dir $(BENCH_OUT_DIR)/medium_$(TS)

bench-medium-compare: build
	@echo "==> Running medium tier head-to-head: $(SOLVERS_COMPARE)"
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite medium --use-cache \
		--solvers $(SOLVERS_COMPARE) \
		--scaled-out-dir $(BENCH_OUT_DIR)/medium_compare_$(TS)

# ── Full tier (~1700 instances, overnight) ───────────────────────────────────

bench-full: build
	@echo "==> Running full tier (discopt, scaled-runner) — this is overnight scale"
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite full --use-cache --subprocess \
		--workers $(BENCH_WORKERS) --mem-limit-mb 0 \
		--scaled-out-dir $(BENCH_OUT_DIR)/full_$(TS)

# ── Baselines + gating ───────────────────────────────────────────────────────

pin-baseline-small: build
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite small --use-cache --subprocess \
		--workers $(BENCH_WORKERS) --mem-limit-mb 0 \
		--scaled-out-dir $(BENCH_OUT_DIR)/small_baseline_$(TS) \
		--pin-baseline

gate-small: build
	$(PYTHON) discopt_benchmarks/run_benchmarks.py \
		--suite small --use-cache --subprocess \
		--workers $(BENCH_WORKERS) --mem-limit-mb 0 \
		--scaled-out-dir $(BENCH_OUT_DIR)/small_gate_$(TS) \
		--gate small

# ── Flag-graduation gate (G1.2) ──────────────────────────────────────────────
# The reusable per-flag validation instrument that lets parked default-OFF
# capabilities flip to default-ON on evidence. Two invocations:
#
#   make graduation-gate      FULL gate — held-out per-flag arm + cert-neutrality
#                             + ledger append. Needs the MINLPLib corpus in
#                             ~/Dropbox/projects/discopt-minlp-benchmark (a corpus
#                             machine only — GitHub CI does NOT have it). This is
#                             the nightly graduation run. Schedule it via cron, e.g.
#                             (a machine with the corpus, from the repo root):
#                               17 4 * * *  cd /path/to/discopt && \
#                                 PYTHONPATH=$PWD/python JAX_PLATFORMS=cpu \
#                                 JAX_ENABLE_X64=1 make graduation-gate >> \
#                                 reports/graduation-nightly.log 2>&1
#
#   make graduation-gate-ci   CI SUBSET — cert-neutrality + incorrect_count over
#                             the VENDORED cert panel only (no corpus, no ledger).
#                             Mirrors .github/workflows/graduation-gate.yml so it
#                             can be reproduced locally.
#
# See docs/dev/flag-graduation-protocol.md for the protocol.

GRAD_FLAGS ?= root_fixpoint,node_reduce,psd_cost_gate,lift_zero_spanning,lift_loose_products
GRAD_N     ?= 100
GRAD_SEED  ?= 0
GRAD_TL    ?= 30

graduation-gate: build
	@echo "==> Flag-graduation gate (FULL): held-out arm + cert-neutrality + ledger"
	PYTHONPATH=$(PROJECT_DIR)/python $(PYTHON) discopt_benchmarks/scripts/graduation_gate.py \
		--flags $(GRAD_FLAGS) --n $(GRAD_N) --seed $(GRAD_SEED) --time-limit $(GRAD_TL)

graduation-gate-ci: build
	@echo "==> Flag-graduation gate (CI SUBSET): cert-neutrality over the vendored panel"
	PYTHONPATH=$(PROJECT_DIR)/python $(PYTHON) discopt_benchmarks/scripts/graduation_gate.py \
		--flags $(GRAD_FLAGS) --ci-subset

# ── Dashboard (local web UI) ─────────────────────────────────────────────────

DASHBOARD_PORT ?= 8765

dashboard:
	@echo "==> Starting dashboard at http://127.0.0.1:$(DASHBOARD_PORT)"
	$(PYTHON) -m discopt_benchmarks.dashboard --port $(DASHBOARD_PORT) --open

# --- Clean --------------------------------------------------------------------

clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf target/debug target/release
	rm -f $(SO_TARGET)
	@echo "==> Clean complete"
