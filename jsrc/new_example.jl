"""
new_example.jl — OWEA optimal design examples.

Run with:  julia --project="." jsrc/new_example.jl

Example 1
---------
D- and A-optimal designs for a 7-factor logistic model:

    logit(P(Y_i = 1 | x_{i1}, ..., x_{i7})) = β_0 + β_1 x_{i1} + ... + β_7 x_{i7}

Design space: full 2^7 = 128 factorial,  x_{ij} ∈ {-1, 1}.
Nominal parameters: β_j = 1 for j = 0, ..., 7.

Example 2
---------
D- and A-optimal designs for a 3-factor continuous logistic model:

    logit(μ_i) = β_0 + β_1 x_{i1} + β_2 x_{i2} + β_3 x_{i3}

(β_0, β_1, β_2, β_3) = (1, -0.5, 0.5, 1),
x_{i1} ∈ [-2, 2],  x_{i2} ∈ [-1, 1],  x_{i3} ∈ [-3, 3],  grid step = 0.01.

Memory note
-----------
At step = 0.01 Example 2 builds 401 × 201 × 601 ≈ 48.4 million grid points.
The info-matrix vector (Vector{Matrix{Float64}}) requires roughly 8 GB of RAM.
Use a coarser step (e.g. 0.05) on machines with less than 10 GB of free RAM.
"""

include("owea.jl")
using .OWEA
using LinearAlgebra, Printf

# ─────────────────────────────────────────────────────────────────────────── #
#  Shared helper                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

"""
    logistic_info(X, β) → Matrix{Float64}  (k²×N)

Fisher information matrices  I(x) = p(x)(1-p(x)) · x xᵀ  for a logistic model.
Returned as a flat k²×N column-major matrix (column n = vec(I_x[n])) so that
OWEASolver can store it without any additional allocation.

Arguments
---------
X : (N × k) design matrix with the intercept column already included.
β : (k,) nominal parameter vector.
"""
function logistic_info(X::Matrix{Float64}, β::Vector{Float64})
    N, k = size(X)
    infos = Matrix{Float64}(undef, k*k, N)
    @inbounds for n in 1:N
        row = view(X, n, :)
        xβ  = clamp(dot(row, β), -500.0, 500.0)
        p   = 1.0 / (1.0 + exp(-xβ))
        s   = sqrt(p * (1.0 - p))
        col = view(infos, :, n)
        idx = 1
        for c in 1:k
            for r in 1:k
                col[idx] = s * row[r] * s * row[c]
                idx += 1
            end
        end
    end
    return infos
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Example 1 — 2^7 binary factorial logistic                                  #
# ─────────────────────────────────────────────────────────────────────────── #

function build_ex1_data()
    levels  = (-1.0, 1.0)
    combos  = vec(collect(Iterators.product(
        levels, levels, levels, levels,
        levels, levels, levels)))           # 128 tuples
    N = length(combos)
    grid = Matrix{Float64}(undef, N, 7)
    X    = Matrix{Float64}(undef, N, 8)
    for (n, c) in enumerate(combos)
        for j in 1:7
            grid[n, j] = c[j]
        end
        X[n, 1] = 1.0
        for j in 1:7
            X[n, j + 1] = c[j]
        end
    end
    return grid, X
end

function print_binary_support(support_points::Matrix{Float64},
                               weights::Vector{Float64})
    n_fac = size(support_points, 2)
    order = sortperm(weights; rev=true)
    header = join(["x$j" for j in 1:n_fac], "  ")
    println("  $header    weight")
    println("  " * "-"^(4 * n_fac + 10))
    for i in order
        row = join(
            [support_points[i, j] > 0.0 ? "+1" : "-1" for j in 1:n_fac],
            "  "
        )
        @printf("  %s   %.6f\n", row, weights[i])
    end
end

function run_example1()
    β    = ones(Float64, 8)
    grid, X = build_ex1_data()
    infos = logistic_info(X, β)

    for (p_val, label) in [(0, "D"), (1, "A")]
        GC.gc()   # free any previous solver's info_stack before allocating a new one
        println("\n=== Example 1: 2^7 logistic — $(label)-optimal (p=$(p_val)) ===")
        solver = OWEASolver(grid, infos, Matrix(1.0I, 8, 8);
                            p=p_val, eps_opt=1e-6, max_outer_iter=300)
        res = solve!(solver)
        @printf("Found in %.3fs, %d iterations, max d = %.3e\n",
                res.elapsed_seconds, res.iterations,
                res.max_directional_derivative)
        if p_val == 0
            d_eff = res.max_directional_derivative < 100 ?
                    exp(-res.max_directional_derivative / 8.0) : 0.0
            @printf("Objective  log|Σ|          = %.6f\n", res.objective_tilde)
            @printf("D-efficiency lower bound   ≥ %.4f\n", d_eff)
        else
            @printf("Objective  Tr(Σ)           = %.6f\n", res.objective_tilde)
        end
        @printf("Number of support points   = %d\n", length(res.weights))
        print_binary_support(res.support_points, res.weights)
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Example 2 — 3-factor continuous logistic                                   #
# ─────────────────────────────────────────────────────────────────────────── #

function build_ex2_data(; step::Float64 = 0.01)
    x1v = range(-2.0, 2.0; step=step)   # 401 values
    x2v = range(-1.0, 1.0; step=step)   # 201 values
    x3v = range(-3.0, 3.0; step=step)   # 601 values
    N = length(x1v) * length(x2v) * length(x3v)
    grid = Matrix{Float64}(undef, N, 3)
    X    = Matrix{Float64}(undef, N, 4)
    n = 0
    for a in x1v, b in x2v, c in x3v
        n += 1
        grid[n, 1] = a;   grid[n, 2] = b;   grid[n, 3] = c
        X[n,    1] = 1.0; X[n,    2] = a;   X[n,    3] = b;   X[n, 4] = c
    end
    return grid, X
end

function print_continuous_support(support_points::Matrix{Float64},
                                   weights::Vector{Float64})
    n_fac = size(support_points, 2)
    order = sortperm(weights; rev=true)
    header = join(["     x$j" for j in 1:n_fac], "   ")
    println("  $header     weight")
    println("  " * "-"^(10 * n_fac + 10))
    for i in order
        row = join([@sprintf("%+9.4f", support_points[i, j]) for j in 1:n_fac], "   ")
        @printf("  %s    %.6f\n", row, weights[i])
    end
end

function run_example2(; step::Float64 = 0.01)
    β = [1.0, -0.5, 0.5, 1.0]

    t0 = time()
    grid, X = build_ex2_data(step=step)
    @printf("\nGrid: %d points  (%.3fs to build)\n", size(grid, 1), time() - t0)

    t1 = time()
    infos = logistic_info(X, β)
    X = nothing; GC.gc()   # free X (1.5 GB) before the solver loop
    @printf("Info matrices: %.2f GB  (built in %.3fs,  size %dx%d)\n",
            sizeof(infos)/1024^3, time() - t1, size(infos, 1), size(infos, 2))

    for (p_val, label) in [(0, "D"), (1, "A")]
        GC.gc()   # free previous solver before allocating a new one
        println("\n=== Example 2: 3-factor logistic — $(label)-optimal (p=$(p_val)) ===")
        solver = OWEASolver(grid, infos, Matrix(1.0I, 4, 4);
                            p=p_val, eps_opt=1e-6, max_outer_iter=300)
        res = solve!(solver)
        @printf("Found in %.3fs, %d iterations, max d = %.3e\n",
                res.elapsed_seconds, res.iterations,
                res.max_directional_derivative)
        if p_val == 0
            d_eff = res.max_directional_derivative < 100 ?
                    exp(-res.max_directional_derivative / 4.0) : 0.0
            @printf("Objective  log|Σ|          = %.6f\n", res.objective_tilde)
            @printf("D-efficiency lower bound   ≥ %.4f\n", d_eff)
        else
            @printf("Objective  Tr(Σ)           = %.6f\n", res.objective_tilde)
        end
        @printf("Number of support points   = %d\n", length(res.weights))
        print_continuous_support(res.support_points, res.weights)
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

function main()
    run_example1()
    run_example2()
end

main()
