"""
new_example.py — OWEA optimal design examples.

Run with:  python new_example.py

Example 1
---------
D- and A-optimal designs for a 7-factor logistic model:

    logit(P(Y_i = 1 | x_{i1}, ..., x_{i7})) = β_0 + β_1 x_{i1} + ... + β_7 x_{i7}

Design space: full 2^7 = 128 factorial,  x_{ij} ∈ {−1, 1}.
Nominal parameters: β_j = 1 for j = 0, ..., 7.

Example 2
---------
D- and A-optimal designs for a 3-factor continuous logistic model:

    logit(μ_i) = β_0 + β_1 x_{i1} + β_2 x_{i2} + β_3 x_{i3}

(β_0, β_1, β_2, β_3) = (1, −0.5, 0.5, 1),
x_{i1} ∈ [−2, 2],  x_{i2} ∈ [−1, 1],  x_{i3} ∈ [−3, 3],  grid step = 0.01.

Memory note
-----------
Example 2 builds a Cartesian product grid of 401 × 201 × 601 ≈ 48.4 million
points.  Info matrices are stored as float32 (~3.1 GB); peak RAM during
construction is roughly 6 GB.  Use a coarser step (e.g. 0.05) on machines
with less than 8 GB of free RAM.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from main import OWEASolver


# ─────────────────────────────────────────────────────────────────────────── #
#  Shared helper                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _logistic_info(X: np.ndarray, beta: np.ndarray,
                   *, float32: bool = False) -> np.ndarray:
    """Vectorised Fisher-information matrices  I(x) = p(1−p) · x xᵀ.

    Parameters
    ----------
    X       : (N, k)  design matrix with the intercept column already included.
    beta    : (k,)    nominal parameter vector.
    float32 : store the result in float32 to halve memory (default False).

    Returns
    -------
    infos : (N, k, k)
    """
    xbeta = np.clip(X @ beta, -500.0, 500.0)          # (N,)
    p = 1.0 / (1.0 + np.exp(-xbeta))                  # sigmoid
    scale = np.sqrt(p * (1.0 - p))                    # (N,)
    f_f64 = scale[:, None] * X                        # (N, k) float64
    if float32:
        f = f_f64.astype(np.float32)
        del f_f64                                      # free float64 copy early
    else:
        f = f_f64
    return f[:, :, None] * f[:, None, :]               # (N, k, k)


# ─────────────────────────────────────────────────────────────────────────── #
#  Example 1 — 2^7 binary factorial logistic                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_ex1_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the 128-point grid and the (128, 8) design matrix."""
    grid = np.array(list(itertools.product([-1.0, 1.0], repeat=7)))  # (128, 7)
    X = np.hstack([np.ones((len(grid), 1)), grid])                    # (128, 8)
    return grid, X


def _print_binary_support(support_points: np.ndarray,
                           weights: np.ndarray) -> None:
    n_fac = support_points.shape[1]
    order = np.argsort(weights)[::-1]
    header = "  ".join(f"x{j + 1}" for j in range(n_fac))
    print(f"  {header}    weight")
    print("  " + "-" * (4 * n_fac + 10))
    for i in order:
        row = "  ".join(
            f"{int(support_points[i, j]):+d}" for j in range(n_fac)
        )
        print(f"  {row}   {weights[i]:.6f}")


def run_example1() -> None:
    """D- and A-optimal designs for the 2^7 logistic model."""
    beta = np.ones(8)                   # β_j = 1 for j = 0, ..., 7
    grid, X = _build_ex1_data()
    infos = _logistic_info(X, beta)     # float64, only 128 points

    for p_val, label in [(0, "D"), (1, "A")]:
        print(f"\n=== Example 1: 2^7 logistic — {label}-optimal (p={p_val}) ===")
        res = OWEASolver(
            grid_points=grid,
            info_matrices=infos,
            g_jacobian=np.eye(8),
            p=p_val,
            eps_opt=1e-6,
            max_outer_iter=300,
        ).solve()

        print(
            f"Found in {res.elapsed_seconds:.3f}s, "
            f"{res.iterations} iterations, "
            f"max d = {res.max_directional_derivative:.3e}"
        )
        if p_val == 0:
            d_eff = (
                math.exp(-res.max_directional_derivative / 8.0)
                if res.max_directional_derivative < 100 else 0.0
            )
            print(f"Objective  log|Σ|          = {res.objective_tilde:.6f}")
            print(f"D-efficiency lower bound   ≥ {d_eff:.4f}")
        else:
            print(f"Objective  Tr(Σ)           = {res.objective_tilde:.6f}")
        print(f"Number of support points   = {len(res.weights)}")
        _print_binary_support(res.support_points, res.weights)


# ─────────────────────────────────────────────────────────────────────────── #
#  Example 2 — 3-factor continuous logistic                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_ex2_data(step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian grid on [−2,2] × [−1,1] × [−3,3] at the given step size.

    Returns
    -------
    grid : (N, 3)  columns  [x1, x2, x3]
    X    : (N, 4)  design matrix  [1, x1, x2, x3]
    """
    x1 = np.arange(-2.0, 2.0 + step / 2, step)   # 401 values
    x2 = np.arange(-1.0, 1.0 + step / 2, step)   # 201 values
    x3 = np.arange(-3.0, 3.0 + step / 2, step)   # 601 values
    g1, g2, g3 = np.meshgrid(x1, x2, x3, indexing="ij")
    grid = np.column_stack([g1.ravel(), g2.ravel(), g3.ravel()])
    X = np.hstack([np.ones((len(grid), 1)), grid])
    return grid, X


def _print_continuous_support(support_points: np.ndarray,
                               weights: np.ndarray) -> None:
    n_fac = support_points.shape[1]
    order = np.argsort(weights)[::-1]
    header = "   ".join(f"     x{j + 1}" for j in range(n_fac))
    print(f"  {header}     weight")
    print("  " + "-" * (10 * n_fac + 10))
    for i in order:
        row = "   ".join(
            f"{support_points[i, j]:+9.4f}" for j in range(n_fac)
        )
        print(f"  {row}    {weights[i]:.6f}")


def run_example2() -> None:
    """D- and A-optimal designs for the 3-factor continuous logistic model."""
    beta = np.array([1.0, -0.5, 0.5, 1.0])

    t0 = time.perf_counter()
    grid, X = _build_ex2_data(step=0.01)
    print(f"\nGrid: {len(grid):,} points  ({time.perf_counter() - t0:.3f}s to build)")

    t1 = time.perf_counter()
    # float32 storage reduces the info-matrix array to ~3.1 GB.
    # The solver promotes to float64 internally for all linear-algebra operations.
    infos = _logistic_info(X, beta, float32=True)
    del X   # free the 1.55 GB design matrix now that infos are built
    print(
        f"Info matrices: {infos.nbytes / 1024**3:.2f} GB  "
        f"(built in {time.perf_counter() - t1:.3f}s,  shape {infos.shape})"
    )

    for p_val, label in [(0, "D"), (1, "A")]:
        print(f"\n=== Example 2: 3-factor logistic — {label}-optimal (p={p_val}) ===")
        res = OWEASolver(
            grid_points=grid,
            info_matrices=infos,
            g_jacobian=np.eye(4),
            p=p_val,
            eps_opt=1e-6,
            max_outer_iter=300,
        ).solve()

        print(
            f"Found in {res.elapsed_seconds:.3f}s, "
            f"{res.iterations} iterations, "
            f"max d = {res.max_directional_derivative:.3e}"
        )
        if p_val == 0:
            d_eff = (
                math.exp(-res.max_directional_derivative / 4.0)
                if res.max_directional_derivative < 100 else 0.0
            )
            print(f"Objective  log|Σ|          = {res.objective_tilde:.6f}")
            print(f"D-efficiency lower bound   ≥ {d_eff:.4f}")
        else:
            print(f"Objective  Tr(Σ)           = {res.objective_tilde:.6f}")
        print(f"Number of support points   = {len(res.weights)}")
        _print_continuous_support(res.support_points, res.weights)


def main() -> None:
    run_example1()
    run_example2()


if __name__ == "__main__":
    main()
