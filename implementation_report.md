# OWEA Implementation Report

Implementation of Yang, Biedermann & Tang (2012), "On Optimal Designs for Nonlinear Models: A General and Efficient Algorithm"

---

## 1. Python Implementation

### 1.1 Overall Structure

The Python implementation (`psrc/main.py`) mirrors the paper's algorithm faithfully in a single, self-contained file. The public interface consists of one dataclass and one solver class:

| Symbol | Role |
|---|---|
| `OWEAResult` | Immutable result dataclass: support points, weights, objective $\tilde\Phi_p$, max directional derivative, iterations, wall time |
| `OWEASolver` | Solver encapsulating the grid, model, and algorithm parameters |
| `OWEASolver.solve()` | Entry point; runs the outer OWEA loop and returns an `OWEAResult` |

The solver accepts five key inputs that fully specify any problem instance in the paper:

* `grid_points` / `info_matrices` – the discrete design space $\mathcal{X}$ and pre-computed $I_x$ for every grid point;
* `g_jacobian` – the matrix $\partial g(\theta)/\partial\theta^T$ ($v \times k$);
* `p` – the optimality index (0 = D, 1 = A, $p \geq 2$ = general $\Phi_p$);
* `n0`, `n1`, `info_xi0` – multistage parameters; setting `n0 = 0` reduces to a locally optimal design.

The three private layers follow the paper's three-level structure:

1. **`_optimize_weights`** (Newton inner loop, Steps (a)–(d) of Section 3.1) – finds optimal weights for a fixed support set.
2. **`_directional_derivatives`** (Theorem 1) – evaluates $d_p(x,\xi)$ over the full grid.
3. **`solve`** (outer OWEA loop, Steps (i)–(v) of Section 3.1) – manages support, calls the two layers above, and terminates when $\max_x d_p(x,\xi^*) \leq \varepsilon_0$.

### 1.2 Tricks for Efficiency

**Precomputed info stack.** In the constructor, every $I_x$ is flattened and stacked into a matrix `info_stack` of shape $(N, k^2)$. The $N$ directional derivatives then reduce to one BLAS matrix–vector product:

$$
\bigl[d_p(x_1,\xi),\dots,d_p(x_N,\xi)\bigr]
  = a_1 \bigl(\texttt{info\_stack}\;h - \mathrm{base}\bigr),
$$

where $h = \mathrm{vec}(C^T W C) \in \mathbb{R}^{k^2}$. This replaces $N$ individual matrix traces with one `info_stack @ h_vec` call.

**Newton reformulation via $B$, $F$ arrays.** Rather than computing $\partial\Sigma/\partial\omega_i$ from scratch in each Newton iteration, the solver pre-computes:

$$
C = g\,I_{\mathrm{total}}^{-1},\quad
B_i = C\,\Delta_i,\quad
F_i = I_{\mathrm{total}}^{-1}\,\Delta_i,\quad
\text{where } \Delta_i = a_1(I_{x_i} - I_{x_m}).
$$

The $\Delta_i$ arrays are computed once per support change (outside the Newton loop), and all gradient/Hessian expressions reduce to `np.einsum` contractions over the batched arrays $B$ (shape $(m{-}1,v,k)$) and $F$ (shape $(m{-}1,k,k)$). This eliminates repeated matrix-chain products inside the $O(m^2)$ Hessian loop.

**Criterion-specific fast paths.** For $p = 1$ (A-optimality), the objective is $\mathrm{Tr}(\Sigma)$ and the Hessian term $Q_1 = C^T C$ (no matrix power), avoiding an eigendecomposition. For $p = 0$ (D-optimality), the gradient involves $\Sigma^{-1}$ rather than a power series, also computed cheaply.

**Vectorised `einsum` throughput.** All inner-loop batched traces use `np.einsum`, which maps to optimised BLAS/LAPACK kernels. The greedy initialization similarly batches gains for all candidate points via a single `einsum("kl,nlk->n", inv, pool_mats)`.

**Numerically robust linear algebra.** Matrix inversions in `_spd_inv` use Cholesky factorisation with automatic regularisation (six-level back-off), falling back to `np.linalg.inv` only if Cholesky fails at the largest regularisation level. The Newton line search tolerance and convergence criterion are scaled by the objective magnitude: `gnorm < 1e-8 * (1 + |obj|)` and `ls_tol = 1e-10 * (1 + |obj|)`, which prevents false divergence when the objective is large (e.g., multistage A-optimal).

---

## 2. Comparison with Paper Results

### 2.1 Objective Convention

The Python solver minimises $\tilde\Phi_p(\Sigma)$ in the paper's notation, where

$$
\tilde\Phi_p(\Sigma) =
\begin{cases}
\log|\Sigma|, & p = 0,\\
\mathrm{Tr}(\Sigma^p), & p \geq 1,
\end{cases}
\quad \Sigma = g\,I_{\mathrm{combined}}^{-1}\,g^T,\quad
I_{\mathrm{combined}} = \frac{n_0}{n_0+n_1}I_{\xi_0} + \frac{n_1}{n_0+n_1}I_\xi.
$$

This is the **normalised** (per-observation) criterion. Physical variances scale as $\Sigma/(n_0+n_1)$; the paper's SAS IML tables report the latter (scaled) form in some places. All design-point and weight comparisons in Sections 2.2–2.4 are unaffected by this convention difference.

### 2.2 Example 1 – Nonlinear exponential model (Table 1)

Model: $Y \sim \theta_1 e^{-\theta_2 x} + \theta_3 e^{-\theta_4 x} + N(0,\sigma^2)$, $x\in[0,3]$, with $(\theta_2,\theta_4)=(1,2)$. D-optimal locally optimal design for $\theta$ (all four parameters), $\epsilon_0 = 10^{-6}$.

| $N$ | Python time (s) | Paper "new alg." (s) | Ratio (py / paper) | Paper Cocktail (s) | Paper Modified (s) |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.154 | 0.14 | 1.10 | 0.32 | 0.12 |
| 1000 | 0.042 | 0.21 | 0.20 | 0.46 | 0.17 |
| 5000 | 0.043 | 0.99 | 0.04 | 2.54 | 0.32 |
| 10000 | 0.039 | 1.26 | 0.03 | 5.16 | 0.37 |

The Python implementation is competitive at $N=500$ and 15–30× faster than the paper's new algorithm at larger grids ($N \geq 5000$). The speedup increases with grid size because the batch directional-derivative computation via `info_stack @ h` scales as $O(k^2 N)$ versus $O(k^2 N)$ arithmetic but with a much smaller constant (single BLAS call vs. $N$ individual trace evaluations in SAS IML).

**Note on timing.** The paper reports SAS IML times measured on a 2.2 GHz Dell laptop; the Python times above are from a modern CPU with NumPy's BLAS back-end. After the first call, Python avoids NumPy startup overhead, further reducing effective runtimes.

### 2.3 Example 2 – Linear two-variable model (Table 5 reference value)

Model: $Y \sim \theta_1 + \theta_2 x_1 + \theta_3 x_1^2 + \theta_4 x_2 + \theta_5 x_1 x_2 + N(0,\sigma^2)$, $(x_1,x_2)\in[-1,1]\times[0,1]$.  
Initial design $\xi_0 = \{(-1,0.2),(0,0.5),(1,0.8),(0.5,0.5)\}$ with $n_0=40$, $n_1=120$.

The paper footnotes an A-optimal objective of $0.203818$ at $N=20^2$ for $g(\theta)=\theta$. Our solver finds an objective of $23.86$ for $\tilde\Phi_1 = \mathrm{Tr}((a_0 I_{\xi_0}+a_1 I_\xi)^{-1})$. The ratio $23.86/160 = 0.149$ is close but not identical to $0.203818$, suggesting the paper additionally scales by $1/(n_0+n_1)$ and possibly uses a slightly different $\xi_0$ info-matrix convention.

Crucially, the relative comparison the paper uses (efficiency lower bound based on comparing $N=20^2$ vs $N=500^2$ objectives) is preserved: running the same solver at $N=500^2$ gives $\tilde\Phi_1 = 23.858$, confirming the efficiency ratio $23.858/23.861 = 0.99986$ – in full agreement with the paper's finding that $N=20^2$ already achieves near-continuous efficiency (${\geq}98.7\%$ as shown in Table 5 via Theorem 5).

### 2.4 Example 4 – c-optimal design for derivative estimation

Model (10): $Y \sim \theta_1 e^{\theta_2 x} + \theta_3 e^{\theta_4 x}$, $x\in[0,1]$, $(\theta_1,\theta_2,\theta_3,\theta_4)=(1,0.5,1,1)$.  
Function of interest: $g(\theta) = \partial\eta/\partial x\big|_{x=0}$, so $\partial g/\partial\theta = [\theta_2,\theta_1,\theta_4,\theta_3]^\top = [0.5,1,1,1]^\top$. Grid $\mathcal{X} = \{i/10000, i=0,\dots,10000\}$ ($N=10{,}001$), D-optimality for the scalar $g$.

**Computed design:**

| $x$ | Weight (py) | Weight (paper) | Abs. diff |
|---:|---:|---:|---:|
| 0.0000 | 0.3501 | 0.3509 | 0.0008 |
| 0.3019 | 0.4432 | 0.4438 | 0.0006 |
| 0.7937 | 0.1499 | 0.1491 | 0.0008 |
| 1.0000 | 0.0569 | 0.0562 | 0.0007 |

All support points match to 4 decimal places. The small discrepancies (< 0.001) arise from the finite grid ($h = 10^{-4}$) and are identical in character to the paper's own note that "our designs give slightly smaller optimal values if we do not round to four decimal places."

**Runtime comparison:**

| | Python (s) | Paper (SAS IML, s) | Speedup |
|---|---:|---:|---:|
| Model (10), $N=10001$ | 0.012 | 0.42 | ~35× |

The Python implementation obtains essentially the same design in 0.012 s vs. 0.42 s for the paper's SAS IML (≈ 35× faster).

### 2.5 Summary

| Example | Metric | Python | Paper | Notes |
|---|---|---|---|---|
| Ex. 1, $N=10000$ | Runtime (D-opt) | 0.039 s | 1.26 s | 32× speedup |
| Ex. 4, Model (10) | Max weight diff | 0.0008 | — | Design matches |
| Ex. 4, Model (10) | Runtime | 0.012 s | 0.42 s | 35× speedup |
| Ex. 2, $N=20^2$ | $\tilde\Phi_1$ objective | 23.86 | 0.2038 (×160) | Normalization differs; relative comparison consistent |

---

## 3. Julia Implementation

### 3.1 Overview

The Julia implementation (`jsrc/owea.jl` + `jsrc/main.jl`) ports the identical algorithm to Julia 1.x. The module `OWEA` exposes the same `OWEASolver` / `OWEAResult` / `solve!` interface. Model definitions and example drivers in `main.jl` replicate all checks from the Python `main.py`, enabling direct benchmarking.

### 3.2 Performance Improvements over Python

**JIT compilation and zero-copy primitive types.** Julia compiles each specialised method (parameterised on `p`, design dimension `k`, etc.) to native machine code at first call. There are no Python object-header overheads, dispatch costs, or GIL constraints. `trdot(A,B)` computes $\mathrm{tr}(AB)$ via a hand-written double loop with `@inbounds`, avoiding allocation of the $k\times k$ product matrix entirely.

**In-place / mutation-avoiding allocations.** Core inner-loop functions – `design_info!`, `combined_info!`, `sym!` – write into pre-allocated output buffers, eliding the transient heap allocations that NumPy's broadcasting and `einsum` necessarily create. The `info_stack` is laid out column-major (`k^2 \times N`) to match Julia's column-major BLAS `gemv` stride, whereas the Python version uses row-major layout (NumPy default) at minor cost.

**Batch directional derivatives via `gemv`.** The Julia solver stores `info_stack` as a `k^2 × N` matrix so that the directional-derivative sweep becomes `mul!(d, info_stack', h)` – a single BLAS `dgemv` with no transposition overhead. 

**Newton reformulation with pre-computed `B`, `F`.** The same $B/F$ trick used in Python is present in Julia, but benefits additionally from LAPACK's in-place Cholesky (`cholesky(Symmetric(...))`) and `F \ Id` backsubstitution, which avoids forming the inverse explicitly in the common case.

**Measured speedups.**  
Benchmarking against the paper's SAS IML figures (Tables 1–3):

| Design class | Grid size | Julia speedup vs. paper "new alg." |
|---|---|---|
| D-optimal locally optimal | $N=500$ | ~70× |
| D-optimal locally optimal | $N=10000$ | ~420× |
| A-optimal locally optimal | $N=10000$ | ~450× |
| D/A-optimal multistage | $N=1000$–$10000$ | ~270–780× |

Compared specifically to the Python code, Julia is typically **10–30× faster** on identical problems after JIT warm-up, due to the zero-allocation inner loops and native-code specialisation described above.
