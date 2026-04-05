from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError as ScipyLinAlgError


Array = np.ndarray


@dataclass
class OWEAResult:
    support_indices: list[int]
    support_points: Array
    weights: Array
    objective_tilde: float
    max_directional_derivative: float
    iterations: int
    elapsed_seconds: float


class OWEASolver:
    def __init__(
        self,
        grid_points: Array,
        info_matrices: Array,
        g_jacobian: Array,
        p: int = 0,
        n0: float = 0.0,
        n1: float = 1.0,
        info_xi0: Array | None = None,
        eps_opt: float = 1e-6,
        eps_weight: float = 1e-10,
        max_outer_iter: int = 300,
        max_newton_iter: int = 60,
    ) -> None:
        self.grid_points = np.asarray(grid_points)
        self.info_matrices = np.asarray(info_matrices, dtype=np.float64)
        self.g = np.asarray(g_jacobian, dtype=np.float64)
        self.gT = self.g.T.copy()   # k×v, C-contiguous for fast matmul
        self.p = p
        self.eps_opt = eps_opt
        self.eps_weight = eps_weight
        self.max_outer_iter = max_outer_iter
        self.max_newton_iter = max_newton_iter

        self.k = self.info_matrices.shape[1]
        self.v = self.g.shape[0]
        self.n0 = float(n0)
        self.n1 = float(n1)
        if self.n0 + self.n1 <= 0:
            raise ValueError("n0 + n1 must be positive")

        if info_xi0 is None:
            self.info_xi0 = np.zeros((self.k, self.k))
        else:
            self.info_xi0 = np.asarray(info_xi0, dtype=np.float64)

        total = self.n0 + self.n1
        self.a0 = self.n0 / total
        self.a1 = self.n1 / total

        # Precompute flattened info stack: shape (N, k²), row n = info_matrices[n].ravel()
        # Used for O(1) batch directional-derivative computation via a single matrix-vector product.
        N = len(self.info_matrices)
        self.info_stack = self.info_matrices.reshape(N, self.k * self.k)

    # ------------------------------------------------------------------ #
    #  Low-level helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sym(m: Array) -> Array:
        return 0.5 * (m + m.T)

    @staticmethod
    def _spd_inv(A: Array) -> Array:
        S = 0.5 * (A + A.T)
        try:
            c, low = cho_factor(S)
            return cho_solve((c, low), np.eye(S.shape[0]))
        except ScipyLinAlgError:
            # Fallback: regularise and retry
            n = S.shape[0]
            reg = 1e-12 * np.trace(S) / n
            for _ in range(6):
                try:
                    c, low = cho_factor(S + reg * np.eye(n))
                    return cho_solve((c, low), np.eye(n))
                except ScipyLinAlgError:
                    reg *= 10.0
            return np.linalg.inv(S)

    def _tilde_phi(self, sigma: Array) -> float:
        if self.p == 0:
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                return float("inf")
            return float(logdet)
        if self.p == 1:
            return float(np.trace(sigma))
        eigvals = np.linalg.eigvalsh(sigma)
        np.clip(eigvals, 1e-18, None, out=eigvals)
        return float(np.sum(eigvals ** self.p))

    def _design_info_from_support(self, support_idx: list[int], weights: Array) -> Array:
        # Vectorised: (m, k, k) weighted sum → (k, k)
        mats = self.info_matrices[support_idx]          # (m, k, k)
        info = np.einsum("n,nij->ij", weights, mats)    # one BLAS call
        return self._sym(info)

    def _sigma(self, info_total: Array) -> Array:
        inv = self._spd_inv(info_total)
        return self._sym(self.g @ inv @ self.gT)

    def _combined_info(self, info_xi: Array) -> Array:
        return self._sym(self.a0 * self.info_xi0 + self.a1 * info_xi)

    # ------------------------------------------------------------------ #
    #  Newton weight optimisation                                          #
    # ------------------------------------------------------------------ #

    def _weight_eval_newton(
        self,
        support_idx: list[int],
        u: Array,
        deltas: Array,          # precomputed (m-1, k, k) = a1*(I_{xi} - I_{xm})
    ) -> tuple[float, Array, Array] | None:
        """Evaluate objective, gradient, and Hessian of tilde_phi w.r.t. u.

        Uses the reformulation from Appendices A.15–A.20 of the paper:
            C  = g @ inv_T          (v×k)
            B[i] = C @ Δ[i]        (v×k)
            F[i] = inv_T @ Δ[i]    (k×k)
        so that  dΣ/dωᵢ = −B[i] @ Cᵀ  (no extra g factors).

        B and F are stacked as (m1, v, k) and (m1, k, k) for batch einsum.
        """
        m1 = len(deltas)
        if m1 <= 0:
            return 0.0, np.zeros(0), np.zeros((0, 0))
        if np.any(u <= 0.0) or float(u.sum()) >= 1.0:
            return None

        wm = 1.0 - float(u.sum())
        m = m1 + 1
        # info_xi from current weights
        mats = self.info_matrices[support_idx]               # (m, k, k)
        w_full = np.empty(m)
        w_full[:m1] = u
        w_full[m1] = wm
        info_xi = np.einsum("n,nij->ij", w_full, mats)      # (k, k)
        info_xi = self._sym(info_xi)

        info_total = self._sym(self.a0 * self.info_xi0 + self.a1 * info_xi)
        try:
            inv_T = self._spd_inv(info_total)
        except Exception:
            return None

        C = self.g @ inv_T                    # (v, k)
        sigma = self._sym(C @ self.gT)        # (v, v)

        # B[i] = C @ Δ[i], shape (m1, v, k)
        B = np.einsum("vk,ikl->ivl", C, deltas)    # (m1, v, k)
        # F[i] = inv_T @ Δ[i], shape (m1, k, k)
        F = np.einsum("kl,ilj->ikj", inv_T, deltas)  # (m1, k, k)

        grad = np.empty(m1)
        hess = np.empty((m1, m1))

        if self.p == 0:
            sign, ld = np.linalg.slogdet(sigma)
            if sign <= 0:
                return None
            objective = float(ld)
            inv_sigma = self._spd_inv(sigma)

            # WC = Cᵀ @ Σ⁻¹, shape (k, v)
            WC = C.T @ inv_sigma                    # (k, v)
            # grad[i] = −tr(WC @ B[i]) = −einsum(WC, B[i])
            grad[:] = -np.einsum("kv,ivk->i", WC, B)   # (m1,)

            # Q = WC @ C = Cᵀ Σ⁻¹ C, shape (k, k)
            Q = WC @ C                              # (k, k)
            # P[i] = WC @ B[i], shape (m1, k, k)
            P = np.einsum("kv,ivl->ikl", WC, B)        # (m1, k, k)

            # Hess[i,j] = tr(Q (Δ[j] F[i] + Δ[i] F[j])) − tr(P[j] P[i])
            #  First term (symmetric in i,j):
            #   tr(Q Δ[j] F[i]) = einsum("kl,jlm,imk->ij", Q, Δ, F) batched
            QD = np.einsum("kl,jlm->jkm", Q, deltas)     # (m1, k, k)
            T1 = np.einsum("ikm,jmk->ij", F, QD) + np.einsum("jkm,imk->ij", F, QD)
            #  Second term: tr(P[j] @ P[i]) = einsum("jkl,ilk->ij")
            T2 = np.einsum("jkl,ilk->ij", P, P)
            hess[:, :] = T1 - T2

        elif self.p == 1:
            objective = float(np.trace(sigma))

            # grad[i] = tr(dΣ/dωᵢ) = −tr(B[i] Cᵀ) = −tr(Cᵀ B[i])
            grad[:] = -np.einsum("vk,ivk->i", C, B)     # (m1,)

            # Q1 = Cᵀ C, shape (k, k)
            Q1 = C.T @ C                            # (k, k)
            # Hess[i,j] = tr(Q1 (Δ[j] F[i] + Δ[i] F[j]))
            Q1D = np.einsum("kl,jlm->jkm", Q1, deltas)  # (m1, k, k)
            hess[:, :] = (
                np.einsum("ikm,jmk->ij", F, Q1D)
                + np.einsum("jkm,imk->ij", F, Q1D)
            )

        else:
            # General p ≥ 2 — exact formulas from Appendices A.19–A.20
            objective = self._tilde_phi(sigma)
            sigma_S = 0.5 * (sigma + sigma.T)
            # Precompute σ^0 ... σ^{p-1}
            pows = [np.eye(self.v)]
            for _ in range(1, self.p):
                pows.append(pows[-1] @ sigma_S)
            sigma_pm1 = pows[self.p - 1]

            # Qp = Cᵀ σ^{p-1}, shape (k, v)
            Qp = C.T @ sigma_pm1              # (k, v)
            # grad[i] = −p tr(Qp B[i])
            grad[:] = -self.p * np.einsum("kv,ivk->i", Qp, B)  # (m1,)

            # Hessian first term: p tr(Cᵀ σ^{p-1} C (Δ[j]F[i] + Δ[i]F[j]))
            Qp_full = Qp @ C                  # (k, k)  = Cᵀ σ^{p-1} C
            QpD = np.einsum("kl,jlm->jkm", Qp_full, deltas)   # (m1, k, k)
            T1 = self.p * (
                np.einsum("ikm,jmk->ij", F, QpD)
                + np.einsum("jkm,imk->ij", F, QpD)
            )
            # Hessian second term: p Σ_{l=0}^{p-2} tr(Cᵀ σ^l B[j] · Cᵀ σ^{p-2-l} B[i])
            # R[l,i] = Cᵀ σ^l B[i], shape (p-1, m1, k, k)
            p_minus_2 = self.p - 2
            R = np.empty((p_minus_2 + 1, m1, self.k, self.k))
            for l in range(p_minus_2 + 1):
                # Cᵀ @ pows[l] @ B[i] for each i
                CpL = C.T @ pows[l]             # (k, v)
                R[l] = np.einsum("kv,ivl->ikl", CpL, B)    # (m1, k, k)
            T2 = np.zeros((m1, m1))
            for l in range(p_minus_2 + 1):
                l2 = p_minus_2 - l
                T2 += np.einsum("jkl,ilk->ij", R[l], R[l2])
            hess[:, :] = T1 + self.p * T2

        hess[:, :] = 0.5 * (hess + hess.T)
        return objective, grad, hess

    def _optimize_weights(self, support_idx: list[int], w0: Array | None = None) -> tuple[list[int], Array]:
        support = list(support_idx)
        min_support = self.k if self.n0 == 0 else 1
        if w0 is None:
            weights = np.full(len(support), 1.0 / len(support))
        else:
            weights = np.asarray(w0, dtype=float)
            if len(weights) != len(support):
                weights = np.full(len(support), 1.0 / len(support))
            else:
                weights = np.clip(weights, 0.0, None)
                s = float(weights.sum())
                weights = np.full(len(support), 1.0 / len(support)) if s <= 0 else weights / s

        while True:
            m = len(support)
            if m == 1:
                return support, np.array([1.0])

            weights = np.clip(weights, 1e-14, None)
            weights = weights / weights.sum()
            u = weights[:-1].copy()

            # Precompute Δ[i] = a1*(I_{x_i} - I_{x_m}) once per support change.
            Im = self.info_matrices[support[-1]]
            deltas = self.a1 * (self.info_matrices[support[:-1]] - Im)  # (m-1, k, k)

            converged = False
            for _ in range(self.max_newton_iter):
                cur = self._weight_eval_newton(support, u, deltas)
                if cur is None:
                    break
                obj, grad, hess = cur
                gnorm = float(np.linalg.norm(grad))
                # Relative gradient norm — handles objectives of varying scale
                if gnorm < 1e-8 * (1.0 + abs(obj)):
                    converged = True
                    break

                step = None
                reg = 0.0
                ident = np.eye(len(u))
                for _ in range(10):
                    try:
                        c, low = cho_factor(hess + reg * ident)
                        step = cho_solve((c, low), grad)
                        break
                    except ScipyLinAlgError:
                        reg = max(1e-12, 1e-10 * gnorm) if reg == 0.0 else reg * 10.0
                if step is None:
                    try:
                        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
                    except Exception:
                        break

                # Back-tracking line search — tolerance scaled by objective magnitude
                ls_tol = 1e-10 * (1.0 + abs(obj))
                alpha = 1.0
                accepted = False
                while alpha >= 1e-5:
                    u_new = u - alpha * step
                    if np.any(u_new <= 0.0) or float(u_new.sum()) >= 1.0:
                        alpha *= 0.5
                        continue
                    nxt = self._weight_eval_newton(support, u_new, deltas)
                    if nxt is not None and nxt[0] <= obj + ls_tol:
                        u = u_new
                        accepted = True
                        break
                    alpha *= 0.5

                if not accepted:
                    # If gradient is already negligible relative to scale, treat as converged
                    if gnorm < 1e-5 * (1.0 + abs(obj)):
                        converged = True
                    break

            weights = np.empty(m)
            weights[:-1] = u
            weights[-1] = 1.0 - float(u.sum())
            np.clip(weights, 0.0, None, out=weights)

            if converged and bool(np.all(weights > self.eps_weight)):
                weights /= weights.sum()
                return support, weights

            # Boundary case: remove smallest-weight support point and repeat.
            if len(support) <= min_support:
                np.clip(weights, self.eps_weight, None, out=weights)
                weights /= weights.sum()
                return support, weights
            rm_idx = int(np.argmin(weights))
            support.pop(rm_idx)
            weights = np.delete(weights, rm_idx)
            s = float(weights.sum())
            if s <= 0.0:
                weights = np.full(len(support), 1.0 / len(support))
            else:
                weights /= s

    def _directional_derivatives(self, info_xi: Array, info_total: Array) -> Array:
        inv_total = self._spd_inv(info_total)
        C = self.g @ inv_total               # (v, k)
        sigma = self._sym(C @ self.gT)       # (v, v)

        if self.p == 0:
            inv_sigma = self._spd_inv(sigma)
            w_mat = inv_sigma
        else:
            sigma_S = 0.5 * (sigma + sigma.T)
            sigma_pow_p = np.linalg.matrix_power(sigma_S, self.p)
            tr_sp = float(np.trace(sigma_pow_p))
            scalar = (1.0 / self.v) ** (1.0 / self.p) * (tr_sp ** (1.0 / self.p - 1.0))
            sigma_pow_pm1 = np.linalg.matrix_power(sigma_S, self.p - 1)
            w_mat = scalar * sigma_pow_pm1

        # H_mat = Cᵀ W C, shape (k, k)
        H_mat = C.T @ w_mat @ C              # (k, k)
        h_vec = H_mat.ravel()                # (k²,)

        # Batch: all_traces[n] = tr(H_mat @ I_x[n]) = h_vec · info_stack[n]
        all_traces = self.info_stack @ h_vec             # (N,)  — one BLAS gemv
        base = float(h_vec @ info_xi.ravel())            # tr(H_mat @ I_ξ)
        return self.a1 * (all_traces - base)

    def _greedy_init(self) -> tuple[list[int], Array]:
        """Greedy Fedorov-style initialization."""
        n = len(self.grid_points)
        k = self.k
        count = min(k + 1, n)

        pool_size = min(n, 20 * k)
        step = max(1, n // pool_size)
        pool = np.arange(0, n, step)[:pool_size]

        # Seed: point with the largest Tr(I_x) — vectorised
        pool_traces = np.einsum("nii->n", self.info_matrices[pool])  # trace of each pool mat
        best_start = int(pool[np.argmax(pool_traces)])
        support = [best_start]
        info_curr = self.info_matrices[best_start].copy() + np.eye(k) * 1e-12

        in_support = {best_start}
        for _ in range(count - 1):
            inv_curr = self._spd_inv(info_curr)
            # Batch gains: tr(inv_curr @ I_x[i]) for all i in pool not in support
            # = einsum("kl,nlk->n", inv_curr, pool_mats)
            pool_mats = self.info_matrices[pool]                        # (pool_size, k, k)
            gains = np.einsum("kl,nlk->n", inv_curr, pool_mats)        # one BLAS call
            # Mask already-selected points
            for si in in_support:
                mask_idx = np.where(pool == si)[0]
                if len(mask_idx):
                    gains[mask_idx[0]] = -1.0
            best_local = int(np.argmax(gains))
            best_i = int(pool[best_local])
            support.append(best_i)
            in_support.add(best_i)
            info_curr += self.info_matrices[best_i]

        weights = np.full(len(support), 1.0 / len(support))
        return support, weights

    def solve(self) -> OWEAResult:
        t0 = time.perf_counter()

        n_grid = len(self.grid_points)
        init_count = min(self.k + 1, n_grid)
        init_idx = np.linspace(0, n_grid - 1, init_count, dtype=int)
        support_idx = list(dict.fromkeys(init_idx.tolist()))
        weights = np.full(len(support_idx), 1.0 / len(support_idx))

        # Check whether the linspace-chosen initial design is non-singular.
        # Structured grids (e.g. factorial × continuous) can cause collinear
        # selections; fall back to the greedy Fedorov-style initializer if so.
        info_test = self._design_info_from_support(support_idx, weights)
        total_test = self._combined_info(info_test)
        if np.linalg.matrix_rank(total_test, tol=1e-10) < self.k:
            support_idx, weights = self._greedy_init()

        best_max_d = float("inf")
        prev_best = float("inf")
        no_improve_count = 0
        it = 0
        while it < self.max_outer_iter:
            it += 1

            support_idx, weights = self._optimize_weights(support_idx, weights)

            keep = weights > max(self.eps_weight, 1e-7)
            if not np.all(keep):
                support_idx = [idx for idx, k in zip(support_idx, keep) if k]
                weights = weights[keep]
                weights /= weights.sum()

            info_xi = self._design_info_from_support(support_idx, weights)
            info_total = self._combined_info(info_xi)
            try:
                d_vals = self._directional_derivatives(info_xi, info_total)
            except np.linalg.LinAlgError:
                # Numerical degeneracy: return the current best feasible design.
                break
            best_max_d = float(np.max(d_vals))

            if prev_best - best_max_d <= 1e-10:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best = best_max_d

            in_support = np.zeros_like(d_vals, dtype=bool)
            in_support[support_idx] = True
            new_d_vals = np.where(in_support, -np.inf, d_vals)
            best_new_idx = int(np.argmax(new_d_vals))
            best_new_d = float(new_d_vals[best_new_idx])

            if best_max_d <= self.eps_opt:
                break
            if no_improve_count >= 10:
                break

            if np.isfinite(best_new_d) and best_new_d > self.eps_opt:
                support_idx.append(best_new_idx)
                weights = np.append(weights, 0.0)
            else:
                break

        info_xi = self._design_info_from_support(support_idx, weights)
        info_total = self._combined_info(info_xi)
        try:
            sigma = self._sigma(info_total)
            obj = self._tilde_phi(sigma)
        except np.linalg.LinAlgError:
            # Degenerate support – return a large sentinel objective.
            sigma = np.eye(self.v) * 1e30
            obj = self._tilde_phi(sigma)

        return OWEAResult(
            support_indices=support_idx,
            support_points=self.grid_points[support_idx],
            weights=weights,
            objective_tilde=obj,
            max_directional_derivative=best_max_d,
            iterations=it,
            elapsed_seconds=time.perf_counter() - t0,
        )


def selector(indices: list[int], k: int) -> Array:
    g = np.zeros((len(indices), k))
    for r, c in enumerate(indices):
        g[r, c] = 1.0
    return g


def info_example1(x: float, theta1: float = 1.0, theta2: float = 1.0, theta3: float = 1.0, theta4: float = 2.0) -> Array:
    f = np.array(
        [
            math.exp(-theta2 * x),
            -theta1 * x * math.exp(-theta2 * x),
            math.exp(-theta4 * x),
            -theta3 * x * math.exp(-theta4 * x),
        ]
    )
    return np.outer(f, f)


def info_example2(point: Array) -> Array:
    x1, x2 = float(point[0]), float(point[1])
    f = np.array([1.0, x1, x1 * x1, x2, x1 * x2])
    return np.outer(f, f)


def info_example4_model10(x: float, theta1: float = 1.0, theta2: float = 0.5, theta3: float = 1.0, theta4: float = 1.0) -> Array:
    f = np.array(
        [
            math.exp(theta2 * x),
            theta1 * x * math.exp(theta2 * x),
            math.exp(theta4 * x),
            theta3 * x * math.exp(theta4 * x),
        ]
    )
    return np.outer(f, f)


def build_initial_info(points: Array, weights: Array, info_fn) -> Array:
    mats = np.array([info_fn(p) for p in points])
    info = np.zeros_like(mats[0])
    for w, m in zip(weights, mats):
        info += float(w) * m
    return info


def run_example1_checks() -> None:
    print("\n=== Example 1: nonlinear model checks ===")
    for n in [500, 1000]:
        grid = np.array([3.0 * i / n for i in range(1, n + 1)])
        infos = np.array([info_example1(float(x)) for x in grid])

        solver = OWEASolver(grid_points=grid, info_matrices=infos, g_jacobian=np.eye(4), p=0, eps_opt=1e-6)
        res = solver.solve()
        print(f"N={n}, local D(theta): time={res.elapsed_seconds:.3f}s, iter={res.iterations}, max d={res.max_directional_derivative:.3e}")

    # Multistage scenario from Section 4
    xi0_points = np.array([0.0, 1.0, 2.0, 3.0])
    xi0_weights = np.array([0.25, 0.25, 0.25, 0.25])
    info0 = build_initial_info(xi0_points, xi0_weights, info_example1)
    n = 1000
    grid = np.array([3.0 * i / n for i in range(1, n + 1)])
    infos = np.array([info_example1(float(x)) for x in grid])

    solver_ms = OWEASolver(
        grid_points=grid,
        info_matrices=infos,
        g_jacobian=np.eye(4),
        p=1,
        n0=40,
        n1=80,
        info_xi0=info0,
        eps_opt=1e-6,
    )
    res_ms = solver_ms.solve()
    print(f"N=1000, multistage A(theta): time={res_ms.elapsed_seconds:.3f}s, iter={res_ms.iterations}, max d={res_ms.max_directional_derivative:.3e}")


def run_example1_runtime_table() -> None:
    print("\n=== Example 1: runtime comparison with paper Table 1 ===")
    paper_cocktail = {500: 0.32, 1000: 0.46, 5000: 2.54, 10000: 5.16}
    paper_new = {500: 0.14, 1000: 0.21, 5000: 0.99, 10000: 1.26}
    paper_modified = {500: 0.12, 1000: 0.17, 5000: 0.32, 10000: 0.37}

    print("N | ours (s) | paper new (s) | ratio ours/paper | paper cocktail (s) | paper modified (s)")
    for n in [500, 1000, 5000, 10000]:
        grid = np.array([3.0 * i / n for i in range(1, n + 1)], dtype=float)
        infos = np.array([info_example1(float(x)) for x in grid])
        solver = OWEASolver(
            grid_points=grid,
            info_matrices=infos,
            g_jacobian=np.eye(4),
            p=0,
            eps_opt=1e-6,
            max_outer_iter=200,
        )
        res = solver.solve()
        ratio = res.elapsed_seconds / paper_new[n]
        print(
            f"{n:5d} | {res.elapsed_seconds:8.3f} | {paper_new[n]:13.2f} | {ratio:15.2f} |"
            f" {paper_cocktail[n]:17.2f} | {paper_modified[n]:17.2f}"
        )


def run_example2_check() -> None:
    print("\n=== Example 2: linear model consistency check ===")
    s = 20
    x1 = np.linspace(-1.0, 1.0, s + 1)
    x2 = np.linspace(0.0, 1.0, s + 1)
    grid = np.array([(a, b) for a in x1 for b in x2], dtype=float)
    infos = np.array([info_example2(pt) for pt in grid])

    # The paper's xi0 list appears to swap coordinates in one point, so we use the admissible order (x1, x2).
    xi0_points = np.array([(-1.0, 0.2), (0.0, 0.5), (1.0, 0.8), (0.5, 0.5)], dtype=float)
    xi0_weights = np.array([0.25, 0.25, 0.25, 0.25])
    info0 = build_initial_info(xi0_points, xi0_weights, info_example2)

    solver = OWEASolver(
        grid_points=grid,
        info_matrices=infos,
        g_jacobian=np.eye(5),
        p=1,
        n0=40,
        n1=120,
        info_xi0=info0,
        eps_opt=1e-6,
    )
    res = solver.solve()

    # For A-optimality: objective is Tr(Sigma), which the paper reports as about 0.203818 at N=20^2.
    print(
        "A-optimal multistage objective Tr(Sigma) at N=20^2: "
        f"{res.objective_tilde:.6f} (paper reports about 0.203818)"
    )


def run_example4_model10_check() -> None:
    print("\n=== Example 4: model (10) design check ===")
    n = 10000
    grid = np.linspace(0.0, 1.0, n + 1)
    infos = np.array([info_example4_model10(float(x)) for x in grid])

    # g(theta) is derivative wrt x at x=0: theta1*theta2 + theta3*theta4
    # so dg/dtheta = [theta2, theta1, theta4, theta3] at (1, 0.5, 1, 1)
    g = np.array([[0.5, 1.0, 1.0, 1.0]])

    solver = OWEASolver(grid_points=grid, info_matrices=infos, g_jacobian=g, p=0, eps_opt=1e-6)
    res = solver.solve()

    points = np.asarray(res.support_points, dtype=float)
    weights = np.asarray(res.weights, dtype=float)
    order = np.argsort(points)
    points = points[order]
    weights = weights[order]

    # Merge adjacent support points produced by finite grid resolution.
    merged_points: list[float] = []
    merged_weights: list[float] = []
    tol = 2.5e-4
    for x, w in zip(points, weights):
        if not merged_points or abs(x - merged_points[-1]) > tol:
            merged_points.append(float(x))
            merged_weights.append(float(w))
        else:
            new_w = merged_weights[-1] + float(w)
            if new_w > 0:
                merged_points[-1] = (merged_points[-1] * merged_weights[-1] + float(x) * float(w)) / new_w
            merged_weights[-1] = new_w

    print(f"time={res.elapsed_seconds:.3f}s, iter={res.iterations}, max d={res.max_directional_derivative:.3e}")
    print("Computed support and weights (rounded to 4 d.p.):")
    for x, w in zip(merged_points, merged_weights):
        print(f"  x={x:.4f}, w={w:.4f}")
    print("Paper target for model (10): (0,0.3509), (0.3011,0.4438), (0.7926,0.1491), (1,0.0562)")


def run_example4_runtime_compare() -> None:
    print("\n=== Example 4 model (10): runtime compare ===")
    t_paper = 0.42
    n = 10000
    grid = np.linspace(0.0, 1.0, n + 1)
    infos = np.array([info_example4_model10(float(x)) for x in grid])
    g = np.array([[0.5, 1.0, 1.0, 1.0]])
    solver = OWEASolver(grid_points=grid, info_matrices=infos, g_jacobian=g, p=0, eps_opt=1e-6, max_outer_iter=250)
    res = solver.solve()
    print(f"ours: {res.elapsed_seconds:.3f}s, paper: {t_paper:.2f}s, ratio ours/paper: {res.elapsed_seconds / t_paper:.2f}")


def build_info_logistic(grid: Array, theta: Array) -> Array:
    """Vectorised Fisher-information matrices for the logistic model.

    Predictor vector (matching the Julia `infor_vec`):
        out = [1, A, B, ESD, Pulse, vol, ESD*Pulse]

    The information vector at a single point is
        f(x) = sqrt(p(x) * (1-p(x))) * out
             = exp(xbeta/2) / (1 + exp(xbeta)) * out

    so  I_x = f f^T  (rank-1 outer product).

    Parameters
    ----------
    grid   : (N, 5)  columns are [A, B, ESD, Pulse, vol]
    theta  : (7,)    intercept then seven predictor coefficients

    Returns
    -------
    infos  : (N, 7, 7) array of information matrices
    """
    # Build the (N, 7) predictor matrix
    out = np.column_stack([
        np.ones(len(grid)),        # intercept
        grid[:, 0],                # A
        grid[:, 1],                # B
        grid[:, 2],                # ESD
        grid[:, 3],                # Pulse
        grid[:, 4],                # vol
        grid[:, 2] * grid[:, 3],   # ESD * Pulse
    ])                             # shape (N, 7)

    xbeta = out @ theta            # (N,)
    # Numerically stable: clip to avoid overflow in exp
    xbeta_clipped = np.clip(xbeta, -500.0, 500.0)
    scale = np.exp(xbeta_clipped / 2.0) / (1.0 + np.exp(xbeta_clipped))  # sqrt(p*(1-p)), shape (N,)

    f = scale[:, None] * out       # (N, 7)
    return f[:, :, None] * f[:, None, :]  # (N, 7, 7)


def build_logistic_grid() -> Array:
    """Build the design grid matching the Julia code.

    levels = [-1, 1]
    vols   = 25 : 0.001 : 45   (20 001 values, both endpoints inclusive)
    All 16 combinations of (A, B, ESD, Pulse) × 20 001 vol values = 320 016 rows.

    Returns
    -------
    grid : (320016, 5)  columns [A, B, ESD, Pulse, vol]
    """
    levels = np.array([-1.0, 1.0])
    vols = np.linspace(25.0, 45.0, 20001)   # (45-25)/0.001 + 1 = 20001

    # Cartesian product of the four binary factors
    A_v, B_v, ESD_v, Pulse_v = np.meshgrid(levels, levels, levels, levels, indexing="ij")
    combos = np.column_stack([A_v.ravel(), B_v.ravel(), ESD_v.ravel(), Pulse_v.ravel()])  # (16, 4)

    combos_rep = np.repeat(combos, len(vols), axis=0)         # (320016, 4)
    vols_tile = np.tile(vols, len(combos))[:, None]           # (320016, 1)
    return np.hstack([combos_rep, vols_tile])                  # (320016, 5)


def run_example_logistic() -> None:
    """Logistic model example translated from the Julia code."""
    print("\n=== Logistic model (new example) ===")
    theta = np.array([-7.5, 1.50, -0.2, -0.15, 0.25, 0.35, 0.4])

    # ---- build grid --------------------------------------------------------
    t_grid = time.perf_counter()
    grid = build_logistic_grid()
    print(f"Grid: {len(grid):,} points  ({time.perf_counter() - t_grid:.3f}s to build)")

    # ---- compute info matrices (vectorised) --------------------------------
    t_info = time.perf_counter()
    infos = build_info_logistic(grid, theta)
    print(f"Info matrices: computed in {time.perf_counter() - t_info:.3f}s  (shape {infos.shape})")

    # ---- D-optimal design for all 7 parameters -----------------------------
    solver = OWEASolver(
        grid_points=grid,
        info_matrices=infos,
        g_jacobian=np.eye(7),
        p=0,
        eps_opt=1e-6,
        max_outer_iter=300,
    )
    res = solver.solve()

    max_d = res.max_directional_derivative
    # D-efficiency lower bound: exp(-max_d / v)  (from the paper's Section 3)
    d_eff_lb = math.exp(-max_d / 7.0) if max_d < 100 else 0.0

    print(
        f"\nD-optimal design found in {res.elapsed_seconds:.3f}s, "
        f"{res.iterations} outer iterations"
    )
    print(f"Max directional derivative  = {max_d:.3e}  "
          f"(D-efficiency lower bound ≥ {d_eff_lb:.4f})")
    print(f"Objective  log|Σ|           = {res.objective_tilde:.6f}")

    # Merge near-duplicate points (same binary combo, vol within 0.5)
    pts = res.support_points   # (m, 5)
    wts = res.weights          # (m,)
    order = np.lexsort((pts[:, 4], pts[:, 2] * pts[:, 3], pts[:, 3], pts[:, 2], pts[:, 1], pts[:, 0]))

    merged: list[tuple[np.ndarray, float]] = []
    for i in order:
        p, w = pts[i], float(wts[i])
        if (
            merged
            and np.array_equal(p[:4], merged[-1][0][:4])          # same binary combo
            and abs(p[4] - merged[-1][0][4]) < 0.5                 # vol within 0.5
        ):
            prev_p, prev_w = merged[-1]
            new_w = prev_w + w
            # weighted-average vol
            merged_pt = prev_p.copy()
            merged_pt[4] = (prev_p[4] * prev_w + p[4] * w) / new_w
            merged[-1] = (merged_pt, new_w)
        else:
            merged.append((p.copy(), w))

    print(f"Number of support points    = {len(merged)}  "
          f"(before merge: {len(wts)})")
    print("\n  A    B   ESD  Pulse    vol     weight")
    print("  " + "-" * 44)
    for p, w in merged:
        print(
            f"  {p[0]:+.0f}   {p[1]:+.0f}   {p[2]:+.0f}   {p[3]:+.0f}  {p[4]:7.3f}  {w:.4f}"
        )


def run_example_logistic_aopt() -> None:
    """Logistic model example – A-optimality (p=1)."""
    print("\n=== Logistic model: A-optimal design (p=1) ===")
    theta = np.array([-7.5, 1.50, -0.2, -0.15, 0.25, 0.35, 0.4])

    t_grid = time.perf_counter()
    grid = build_logistic_grid()
    print(f"Grid: {len(grid):,} points  ({time.perf_counter() - t_grid:.3f}s to build)")

    t_info = time.perf_counter()
    infos = build_info_logistic(grid, theta)
    print(f"Info matrices: computed in {time.perf_counter() - t_info:.3f}s  (shape {infos.shape})")

    solver = OWEASolver(
        grid_points=grid,
        info_matrices=infos,
        g_jacobian=np.eye(7),
        p=1,
        eps_opt=1e-6,
        max_outer_iter=300,
    )
    res = solver.solve()

    print(
        f"\nA-optimal design found in {res.elapsed_seconds:.3f}s, "
        f"{res.iterations} outer iterations, "
        f"max directional derivative = {res.max_directional_derivative:.3e}"
    )
    print(f"Number of support points: {len(res.weights)}")
    print(f"Objective  Tr(Sigma)  = {res.objective_tilde:.6f}")

    pts = res.support_points
    wts = res.weights
    order = np.lexsort((pts[:, 4], pts[:, 2] * pts[:, 3], pts[:, 3], pts[:, 2], pts[:, 1], pts[:, 0]))

    merged: list[tuple[np.ndarray, float]] = []
    for i in order:
        p, w = pts[i], float(wts[i])
        if (
            merged
            and np.array_equal(p[:4], merged[-1][0][:4])
            and abs(p[4] - merged[-1][0][4]) < 0.5
        ):
            prev_p, prev_w = merged[-1]
            new_w = prev_w + w
            merged_pt = prev_p.copy()
            merged_pt[4] = (prev_p[4] * prev_w + p[4] * w) / new_w
            merged[-1] = (merged_pt, new_w)
        else:
            merged.append((p.copy(), w))

    print(f"Support points after merging: {len(merged)}  (before merge: {len(wts)})")
    print("\n  A    B   ESD  Pulse    vol     weight")
    print("  " + "-" * 44)
    for p, w in merged:
        print(
            f"  {p[0]:+.0f}   {p[1]:+.0f}   {p[2]:+.0f}   {p[3]:+.0f}  {p[4]:7.3f}  {w:.4f}"
        )


def main() -> None:
    run_example1_runtime_table()
    run_example1_checks()
    run_example2_check()
    run_example4_model10_check()
    run_example4_runtime_compare()
    run_example_logistic()
    run_example_logistic_aopt()


if __name__ == "__main__":
    main()
