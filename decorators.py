'''
Lightweight decorator markers for MLX.

These decorators are simple wrappers. MLX parses your experiment file's AST,
so the wrappers do not alter runtime behavior. They only:
  - Preserve function identity
  - Accept positional or keyword arguments (to match examples)
  - Attach attributes for human readability and potential introspection

Usage
-----
@trial(order, values) or @trial(order=..., values=...)
  - order: int (execution order among trials)
  - values: dict[str, list] mapping parameter name -> list of values.

@verify(["trial1", "trial2"]) or @verify(trial_names=[...])
  - trial_names: list[str] of trial function names whose results to verify.

@setup(order: int | None = None)
  - Runs once before trials to produce a run-level context (JSON-serializable dict).
  - The runner will expose this context to trials/verification (details in docs).

Verification return values
--------------------------
Verification functions may return:
  - VerificationResult enum (preferred), or
  - one of: "supports", "refutes", "inconclusive", "invalid", "continue", or
  - a boolean (True -> SUPPORTS, False -> REFUTES), or
  - None (treated as INCONCLUSIVE).
'''

from __future__ import annotations

from enum import Enum
import functools
from typing import Any, Callable, Dict, List, Optional


class VerificationResult(Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"  # Bad data/failed precondition
    CONTINUE = "continue"  # For intermediate verifications


def trial(
    order: Optional[int] = None,
    values: Optional[Dict[str, List[Any]]] = None,
):
    """Mark a function as a trial.

    Accepts positional or keyword args. Examples:
      @trial(1, {"a": [1, 2], "b": [3, 4]})
      @trial(order=1, values={"a": [1, 2], "b": [3, 4]})
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Intentionally no behavior changes — MLX reconstructs and executes code separately.
            return func(*args, **kwargs)

        # Attach helpful attributes for humans/future tooling
        setattr(wrapper, "__mlx_trial__", True)
        setattr(wrapper, "__mlx_order__", order)
        setattr(wrapper, "__mlx_values__", values)
        return wrapper

    return decorator


def verify(trial_names: Optional[List[str]] = None):
    """Mark a function as a verification step.

    Accepts positional or keyword args. Examples:
      @verify(["trial1", "trial2"])  # positional
      @verify(trial_names=["trial1", "trial2"])  # keyword
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Intentionally no behavior changes — MLX reconstructs and executes code separately.
            return func(*args, **kwargs)

        setattr(wrapper, "__mlx_verify__", True)
        setattr(wrapper, "__mlx_verify_trials__", trial_names or [])
        return wrapper

    return decorator


def setup(order: Optional[int] = None):
    """Mark a function as a one-time setup step for the run.

    The function should return a JSON-serializable dict that represents the
    run-level context (e.g., seeds, paths, configuration). The runner will run
    setup before any trials and make the context available to trials and
    verification (see documentation for specifics and limitations).

    Example:
        @setup(order=1)
        def init():
            return {"seed": 42, "numpy_seed": 42, "torch_seed": 42}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # No behavior change at definition site.
            return func(*args, **kwargs)

        setattr(wrapper, "__mlx_setup__", True)
        setattr(wrapper, "__mlx_order__", order)
        return wrapper

    return decorator