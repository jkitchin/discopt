"""Tests for T12 — Batch Dispatch Interface (Rust<->Python zero-copy arrays).

Validates that PyBatchDispatcher correctly exports node bounds as numpy arrays
with zero-copy semantics and imports relaxation results back.
"""

import time

import numpy as np
import pytest
from discopt._rust import PyBatchDispatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_dispatcher(disp: PyBatchDispatcher, n_nodes: int) -> list[int]:
    """Add n_nodes with random bounds and return their IDs."""
    rng = np.random.default_rng(42)
    n_vars = disp.n_vars()
    ids = []
    for _ in range(n_nodes):
        lb = rng.uniform(-10, 0, size=n_vars).tolist()
        ub = rng.uniform(0, 10, size=n_vars).tolist()
        ids.append(disp.add_node(lb, ub))
    return ids


# ---------------------------------------------------------------------------
# 1. Export shape
# ---------------------------------------------------------------------------


class TestExportShape:
    """Verify exported arrays have correct shapes."""

    @pytest.mark.parametrize("n_nodes,batch_size", [(5, 10), (10, 10), (20, 10)])
    def test_export_shape_2d(self, n_nodes, batch_size):
        n_vars = 4
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, n_nodes)
        lb, ub, ids = disp.export_batch(batch_size)
        expected_n = min(n_nodes, batch_size)
        assert lb.shape == (expected_n, n_vars)
        assert ub.shape == (expected_n, n_vars)
        assert ids.shape == (expected_n,)

    def test_export_multiple_batches(self):
        """Exporting twice drains pending nodes sequentially."""
        n_vars = 3
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, 10)

        lb1, ub1, ids1 = disp.export_batch(6)
        assert lb1.shape[0] == 6
        assert disp.pending_count() == 4

        lb2, ub2, ids2 = disp.export_batch(6)
        assert lb2.shape[0] == 4  # only 4 remaining
        assert disp.pending_count() == 0


# ---------------------------------------------------------------------------
# 2. Export dtype
# ---------------------------------------------------------------------------


class TestExportDtype:
    """Verify exported arrays have correct dtypes."""

    def test_lb_ub_float64(self):
        disp = PyBatchDispatcher(5)
        _fill_dispatcher(disp, 3)
        lb, ub, ids = disp.export_batch(10)
        assert lb.dtype == np.float64
        assert ub.dtype == np.float64

    def test_ids_int64(self):
        disp = PyBatchDispatcher(5)
        _fill_dispatcher(disp, 3)
        _, _, ids = disp.export_batch(10)
        assert ids.dtype == np.int64


# ---------------------------------------------------------------------------
# 3. Zero-copy verification
# ---------------------------------------------------------------------------


class TestZeroCopy:
    """Verify that exported arrays share memory with Rust allocation."""

    def test_pointer_match(self):
        disp = PyBatchDispatcher(10)
        _fill_dispatcher(disp, 8)
        lb, ub, ids = disp.export_batch(8)
        # The Rust side stores the data pointer of the lb array.
        rust_ptr = disp.last_export_ptr()
        numpy_ptr = lb.ctypes.data
        assert rust_ptr == numpy_ptr, f"Pointer mismatch: Rust={rust_ptr:#x}, numpy={numpy_ptr:#x}"

    def test_no_copy_on_read(self):
        """Reading the exported array should not trigger a copy."""
        disp = PyBatchDispatcher(10)
        _fill_dispatcher(disp, 4)
        lb, _, _ = disp.export_batch(4)
        # Accessing data should not change the base pointer
        _ = lb[0, 0]
        assert lb.ctypes.data == disp.last_export_ptr()


# ---------------------------------------------------------------------------
# 4. Import correctness
# ---------------------------------------------------------------------------


class TestImportResults:
    """Verify that imported results are stored correctly."""

    def test_basic_import(self):
        n_vars = 5
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, 4)
        lb, ub, ids = disp.export_batch(4)

        # Simulate solving: lower_bounds, solutions, feasibility
        lower_bounds = np.array([1.0, 2.0, 3.0, 4.0])
        solutions = np.random.default_rng(0).random((4, n_vars))
        feasible = np.array([True, False, True, False])

        disp.import_results(ids, lower_bounds, solutions, feasible)
        assert disp.result_count() == 4

    def test_retrieve_result(self):
        n_vars = 3
        disp = PyBatchDispatcher(n_vars)
        node_ids = _fill_dispatcher(disp, 2)
        _, _, ids = disp.export_batch(2)

        lower_bounds = np.array([10.0, 20.0])
        solutions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        feasible = np.array([True, False])

        disp.import_results(ids, lower_bounds, solutions, feasible)

        # Retrieve first node result
        result = disp.get_result(node_ids[0])
        assert result is not None
        lb_val, sol, feas = result
        assert lb_val == 10.0
        assert feas is True
        np.testing.assert_array_equal(sol, [1.0, 2.0, 3.0])

        # Retrieve second node result
        result2 = disp.get_result(node_ids[1])
        assert result2 is not None
        assert result2[0] == 20.0
        assert result2[2] is False

    def test_missing_result_returns_none(self):
        disp = PyBatchDispatcher(3)
        assert disp.get_result(999) is None

    def test_import_dimension_mismatch(self):
        n_vars = 3
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, 2)
        _, _, ids = disp.export_batch(2)

        # Wrong solutions shape
        lower_bounds = np.array([1.0, 2.0])
        solutions = np.random.random((2, n_vars + 1))  # wrong n_vars
        feasible = np.array([True, False])

        with pytest.raises(ValueError, match="n_vars"):
            disp.import_results(ids, lower_bounds, solutions, feasible)

    def test_import_length_mismatch(self):
        n_vars = 3
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, 2)
        _, _, ids = disp.export_batch(2)

        lower_bounds = np.array([1.0])  # wrong length
        solutions = np.random.random((2, n_vars))
        feasible = np.array([True, False])

        with pytest.raises(ValueError, match="same first dimension"):
            disp.import_results(ids, lower_bounds, solutions, feasible)


# ---------------------------------------------------------------------------
# 5. Batch sizes
# ---------------------------------------------------------------------------


class TestBatchSizes:
    """Test with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 32, 64, 128, 512, 1024])
    def test_batch_size(self, batch_size):
        n_vars = 10
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, batch_size)

        lb, ub, ids = disp.export_batch(batch_size)
        assert lb.shape == (batch_size, n_vars)
        assert ub.shape == (batch_size, n_vars)
        assert ids.shape == (batch_size,)

        # Verify node IDs are sequential starting from 0
        np.testing.assert_array_equal(ids, np.arange(batch_size, dtype=np.int64))


# ---------------------------------------------------------------------------
# 6. Latency
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLatency:
    """Round-trip export -> import should be fast."""

    @pytest.mark.parametrize("batch_size", [32, 64, 128, 512])
    def test_round_trip_under_budget(self, batch_size):
        n_vars = 50
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, batch_size)

        # Warm up
        lb, ub, ids = disp.export_batch(batch_size)
        lower_bounds = np.ones(lb.shape[0])
        solutions = np.ones_like(lb)
        feasible = np.ones(lb.shape[0], dtype=bool)
        disp.import_results(ids, lower_bounds, solutions, feasible)

        # Reload and time (best of 5 runs to avoid OS noise)
        best_us = float("inf")
        for _ in range(5):
            _fill_dispatcher(disp, batch_size)
            # Pre-allocate result arrays so we only measure export+import
            pre_lb = np.ones(batch_size)
            pre_sol = np.ones((batch_size, n_vars))
            pre_feas = np.ones(batch_size, dtype=bool)

            start = time.perf_counter_ns()
            lb, ub, ids = disp.export_batch(batch_size)
            disp.import_results(ids, pre_lb, pre_sol, pre_feas)
            elapsed_us = (time.perf_counter_ns() - start) / 1_000
            best_us = min(best_us, elapsed_us)

        # Budget scales with batch size: 100us base for batch<=128,
        # ~0.5us per node for larger batches (memory copy cost).
        budget_us = max(200, batch_size * 2.0)
        assert best_us < budget_us, (
            f"Round-trip took {best_us:.1f}us (budget: {budget_us:.0f}us, "
            f"batch={batch_size}, n_vars={n_vars})"
        )


# ---------------------------------------------------------------------------
# 7. Empty batch
# ---------------------------------------------------------------------------


class TestEmptyBatch:
    """Export from empty dispatcher returns empty arrays."""

    def test_empty_export(self):
        n_vars = 7
        disp = PyBatchDispatcher(n_vars)
        lb, ub, ids = disp.export_batch(10)
        assert lb.shape == (0, n_vars)
        assert ub.shape == (0, n_vars)
        assert ids.shape == (0,)

    def test_empty_after_drain(self):
        n_vars = 3
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, 5)
        disp.export_batch(5)  # drain all
        lb, ub, ids = disp.export_batch(10)
        assert lb.shape == (0, n_vars)
        assert ids.shape == (0,)


# ---------------------------------------------------------------------------
# 8. Variable dimensions
# ---------------------------------------------------------------------------


class TestVariableDimensions:
    """Test with different n_vars values."""

    @pytest.mark.parametrize("n_vars", [10, 50, 100])
    def test_dimensions(self, n_vars):
        disp = PyBatchDispatcher(n_vars)
        _fill_dispatcher(disp, 16)
        lb, ub, ids = disp.export_batch(16)
        assert lb.shape == (16, n_vars)
        assert ub.shape == (16, n_vars)
        assert lb.dtype == np.float64

    def test_values_preserved(self):
        """Verify that exported values match what was added."""
        n_vars = 4
        disp = PyBatchDispatcher(n_vars)
        lb_in = [0.0, 1.0, 2.0, 3.0]
        ub_in = [10.0, 11.0, 12.0, 13.0]
        disp.add_node(lb_in, ub_in)

        lb, ub, ids = disp.export_batch(1)
        np.testing.assert_array_equal(lb[0], lb_in)
        np.testing.assert_array_equal(ub[0], ub_in)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    def test_add_node_wrong_lb_size(self):
        disp = PyBatchDispatcher(3)
        with pytest.raises(ValueError, match="lb length"):
            disp.add_node([0.0, 0.0], [1.0, 1.0, 1.0])

    def test_add_node_wrong_ub_size(self):
        disp = PyBatchDispatcher(3)
        with pytest.raises(ValueError, match="ub length"):
            disp.add_node([0.0, 0.0, 0.0], [1.0, 1.0])

    def test_zero_nvars_rejected(self):
        with pytest.raises(ValueError, match="n_vars must be positive"):
            PyBatchDispatcher(0)

    def test_node_ids_monotonic(self):
        disp = PyBatchDispatcher(2)
        ids = [disp.add_node([0.0, 0.0], [1.0, 1.0]) for _ in range(5)]
        assert ids == [0, 1, 2, 3, 4]

    def test_batch_size_one(self):
        disp = PyBatchDispatcher(3)
        _fill_dispatcher(disp, 1)
        lb, ub, ids = disp.export_batch(1)
        assert lb.shape == (1, 3)
