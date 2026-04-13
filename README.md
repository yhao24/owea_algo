# OWEA Algorithmic Implementation

Implementation of the **Optimal Weights Exchange Algorithm (OWEA)** for finding optimal designs in nonlinear models, as described in:
> Yang, M., Biedermann, S., & Tang, E. (2012). *On Optimal Designs for Nonlinear Models: A General and Efficient Algorithm*. Journal of the American Statistical Association.

This repository contains both Python and Julia implementations of the algorithm, covering locally optimal and multistage designs under $\Phi_p$-optimality criteria.

## 🚀 Key Features

- **General Optimality:** Supports $D$-optimality ($p=0$), $A$-optimality ($p=1$), and general $\Phi_p$-optimality for integer $p \ge 2$.
- **Flexible Objectives:** Handles the full parameter vector $\theta$, sub-vectors, or any differentiable function $g(\theta)$.
- **Multistage Support:** Efficiently identifies second-stage designs $\xi$ given an initial design $\xi_0$ and sample sizes $n_0, n_1$.
- **High Performance:** 
    - **Python:** Optimized using `numpy` vectorization and `einsum` for batch Hessian/gradient calculations.
    - **Julia:** Further optimized with zero-allocation inner loops and BLAS/LAPACK kernels, achieving up to 780x speedup over the original paper's SAS IML timings.

## 📁 Project Structure

```
.
├── psrc/
│   └── main.py        # Python OWEA solver and example drivers
├── jsrc/
│   ├── owea.jl        # Core Julia OWEA module
│   └── main.jl        # Julia example drivers (matches main.py)
├── OWEA-Algorithm-paper.md   # Markdown version of the source paper
└── implementation_report.md  # Detailed implementation & performance report
```

## 🛠️ Performance Summary

The implementations significantly outperform the SAS IML code used in the original paper by leveraging modern BLAS-backed vectorization and Newton-method reformulations.

| Example | Grid Size ($N$) | Python Speedup | Julia Speedup |
| :--- | :--- | :--- | :--- |
| Ex 1 (D-opt) | 10,000 | ~32x | ~420x |
| Ex 1 (A-MS)  | 1,000  | ~20x | ~270x |
| Ex 4 (c-opt) | 10,001 | ~35x | ~400x |

*Note: Speedup is relative to the "New Algorithm" timings reported in the 2012 paper.*

## 📖 Usage

### Python
Requires `numpy` and `scipy`.
```powershell
cd psrc
python main.py
```

### Julia
Built for Julia 1.12.5
```powershell
julia --project="." jsrc/main.jl
```

## 📝 Implementation Details
For a deep dive into the algorithmic optimizations (such as the $B/F$ array reformulation for Newton steps and the precomputed info stack for directional derivatives), see [implementation_report.md](implementation_report.md).
