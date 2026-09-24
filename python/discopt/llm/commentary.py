"""
Streaming LLM commentary during Branch & Bound solve.

Feature 2C: Periodic async LLM calls with solve state to produce
human-readable messages about B&B progress.

Commentary is purely decorative — it never affects solver decisions.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SolveCommentator:
    """Generates LLM commentary during a B&B solve.

    Runs LLM calls in a background thread to avoid blocking the solve loop.
    Commentary is queued and consumed by the streaming interface.

    Parameters
    ----------
    model_summary : str
        Text summary of the model being solved.
    llm_model : str, optional
        LLM model string.
    interval : float, default 10.0
        Minimum seconds between commentary requests.
    """

    def __init__(
        self,
        model_summary: str,
        llm_model: str | None = None,
        interval: float = 10.0,
    ):
        self._model_summary = model_summary
        self._llm_model = llm_model
        self._interval = interval
        self._last_comment_time = 0.0
        self._pending_message: Optional[str] = None
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._enabled = True

        # Check LLM availability once
        try:
            from discopt.llm import is_available

            self._enabled = is_available()
        except Exception:
            self._enabled = False

    def maybe_comment(
        self,
        elapsed: float,
        incumbent: float | None,
        lower_bound: float,
        gap: float | None,
        node_count: int,
        open_nodes: int,
        iteration: int,
    ) -> str | None:
        """Check if commentary should be generated and return it.

        Called from the B&B loop. If enough time has passed since the
        last commentary and a new message is available, returns it.
        If no message is ready, may trigger an async LLM call and
        return None.

        Parameters
        ----------
        elapsed : float
            Wall-clock time since solve start.
        incumbent : float or None
            Best feasible objective found so far.
        lower_bound : float
            Current global lower bound.
        gap : float or None
            Current relative optimality gap.
        node_count : int
            Total nodes explored.
        open_nodes : int
            Nodes remaining in the tree.
        iteration : int
            B&B iteration count.

        Returns
        -------
        str or None
            Commentary message, or None if nothing new.
        """
        if not self._enabled:
            return None

        # Check if there's a pending message from a background call
        with self._lock:
            if self._pending_message is not None:
                msg = self._pending_message
                self._pending_message = None
                return msg

        # Rate-limit commentary
        now = time.monotonic()
        if now - self._last_comment_time < self._interval:
            return None

        self._last_comment_time = now

        # Build state snapshot
        state = {
            "elapsed": f"{elapsed:.1f}s",
            "incumbent": f"{incumbent:.6g}" if incumbent is not None else "none",
            "lower_bound": f"{lower_bound:.6g}",
            "gap": f"{gap:.2%}" if gap is not None else "unknown",
            "nodes_explored": node_count,
            "nodes_open": open_nodes,
            "iteration": iteration,
        }

        # Fire background LLM call
        thread = threading.Thread(
            target=self._generate_comment,
            args=(state,),
            daemon=True,
        )
        thread.start()
        return None

    def get_root_comment(
        self,
        objective: float | None,
        lower_bound: float,
        gap: float | None,
    ) -> str | None:
        """Generate commentary after the root node solve.

        This is a synchronous call since the root node result is
        especially informative.

        Returns
        -------
        str or None
            Commentary about the root relaxation, or None on failure.
        """
        if not self._enabled:
            return None

        state = {
            "event": "root_node_solved",
            "root_objective": (f"{objective:.6g}" if objective is not None else "infeasible"),
            "root_lower_bound": f"{lower_bound:.6g}",
            "root_gap": f"{gap:.2%}" if gap is not None else "unknown",
        }

        return self._generate_comment_sync(state)

    def get_new_incumbent_comment(
        self,
        old_objective: float | None,
        new_objective: float,
        gap: float | None,
        node_count: int,
    ) -> str | None:
        """Generate commentary when a new incumbent is found.

        Returns
        -------
        str or None
            Commentary about the improvement, or None on failure.
        """
        if not self._enabled:
            return None

        state = {
            "event": "new_incumbent",
            "previous_best": (f"{old_objective:.6g}" if old_objective is not None else "none"),
            "new_best": f"{new_objective:.6g}",
            "gap": f"{gap:.2%}" if gap is not None else "unknown",
            "nodes_explored": node_count,
        }

        return self._generate_comment_sync(state)

    def _generate_comment(self, state: dict):
        """Background thread: generate commentary and store it."""
        try:
            msg = self._generate_comment_sync(state)
            if msg:
                with self._lock:
                    self._pending_message = msg
        except Exception as e:
            logger.debug("Commentary generation failed: %s", e)

    def _generate_comment_sync(self, state: dict) -> str | None:
        """Synchronous LLM call to generate a commentary message."""
        try:
            from discopt.llm.provider import complete, resolve_timeout
        except ImportError:
            return None

        self._history.append(state)

        # Keep history compact
        recent = self._history[-5:]

        prompt = (
            "You are providing live commentary on an optimization solve. "
            "Give a brief (1-2 sentence) update about the solve progress. "
            "Be specific about numbers. Use plain language a non-expert "
            "can understand.\n\n"
            f"Model: {self._model_summary}\n\n"
            f"Solve history (most recent last):\n"
        )
        for h in recent:
            prompt += f"  {h}\n"
        prompt += "\nProvide a brief commentary message:"

        try:
            raw = complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._llm_model,
                max_tokens=150,
                timeout=resolve_timeout(default=5.0),
            )
            return raw.strip() if raw else None
        except Exception:
            return None
