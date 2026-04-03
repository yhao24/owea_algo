"""
    main.jl — Examples matching the Python main.py and the OWEA paper

Run with:   julia --project="." jsrc/main.jl
"""

include("owea.jl")
using .OWEA
using LinearAlgebra, Printf

# ─────────────────────────────────────────────────────────────────────────── #
#  Utility                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

"""Selector matrix: rows pick indices out of k parameters."""
function selector(indices::AbstractVector{Int}, k::Int)
    g = zeros(length(indices), k)
    for (r, c) in enumerate(indices)
        g[r, c] = 1.0
    end
    return g
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Example 1 – nonlinear model  Y ~ θ₁e^{-θ₂x} + θ₃e^{-θ₄x}               #
# ─────────────────────────────────────────────────────────────────────────── #

function info_example1(x::Float64;
                       θ1=1.0, θ2=1.0, θ3=1.0, θ4=2.0)
    f = [exp(-θ2 * x),
         -θ1 * x * exp(-θ2 * x),
         exp(-θ4 * x),
         -θ3 * x * exp(-θ4 * x)]
    return f * f'
end

function build_initial_info(points, weights, info_fn)
    mats = [info_fn(Float64(p)) for p in points]
    info = zeros(size(mats[1]))
    for (w, m) in zip(weights, mats)
        info .+= w .* m
    end
    return info
end

function run_example1_checks()
    println("\n=== Example 1: nonlinear model checks ===")
    for n in [500, 1000]
        grid = [3.0 * i / n for i in 1:n]
        infos = [info_example1(x) for x in grid]
        grid_mat = reshape(grid, :, 1)

        solver = OWEASolver(grid_mat, infos, Matrix(1.0I, 4, 4);
                            p=0, eps_opt=1e-6)
        res = solve!(solver)
        @printf("N=%d, local D(θ): time=%.3fs, iter=%d, max d=%.3e\n",
                n, res.elapsed_seconds, res.iterations,
                res.max_directional_derivative)
    end

    # Multistage scenario
    xi0_pts = [0.0, 1.0, 2.0, 3.0]
    xi0_w   = [0.25, 0.25, 0.25, 0.25]
    info0   = build_initial_info(xi0_pts, xi0_w, info_example1)

    n = 1000
    grid = [3.0 * i / n for i in 1:n]
    infos = [info_example1(x) for x in grid]
    grid_mat = reshape(grid, :, 1)

    solver = OWEASolver(grid_mat, infos, Matrix(1.0I, 4, 4);
                        p=1, n0=40.0, n1=80.0, info_xi0=info0, eps_opt=1e-6)
    res = solve!(solver)
    @printf("N=1000, multistage A(θ): time=%.3fs, iter=%d, max d=%.3e\n",
            res.elapsed_seconds, res.iterations,
            res.max_directional_derivative)
end

function run_example1_runtime_table()
    println("\n=== Example 1: runtime comparison with paper Table 1 ===")
    paper_new = Dict(500 => 0.14, 1000 => 0.21, 5000 => 0.99, 10000 => 1.26)
    paper_cocktail = Dict(500 => 0.32, 1000 => 0.46, 5000 => 2.54, 10000 => 5.16)

    @printf("%-6s | %8s | %13s | %16s | %18s\n",
            "N", "ours (s)", "paper new (s)", "ratio ours/paper", "paper cocktail (s)")
    println("-"^75)
    for n in [500, 1000, 5000, 10000]
        grid = [3.0 * i / n for i in 1:n]
        infos = [info_example1(x) for x in grid]
        grid_mat = reshape(grid, :, 1)

        solver = OWEASolver(grid_mat, infos, Matrix(1.0I, 4, 4);
                            p=0, eps_opt=1e-6, max_outer_iter=200)
        res = solve!(solver)
        ratio = res.elapsed_seconds / paper_new[n]
        @printf("%5d  | %8.3f | %13.2f | %15.2f  | %17.2f\n",
                n, res.elapsed_seconds, paper_new[n], ratio, paper_cocktail[n])
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Example 2 – linear model  (x₁,x₂) ∈ [-1,1]×[0,1]                         #
# ─────────────────────────────────────────────────────────────────────────── #

function info_example2(x1::Float64, x2::Float64)
    f = [1.0, x1, x1*x1, x2, x1*x2]
    return f * f'
end

function run_example2_check()
    println("\n=== Example 2: linear model consistency check ===")
    s = 20
    x1v = range(-1.0, 1.0, length=s+1)
    x2v = range(0.0, 1.0, length=s+1)

    grid = Matrix{Float64}(undef, (s+1)^2, 2)
    infos = Vector{Matrix{Float64}}(undef, (s+1)^2)
    idx = 0
    for a in x1v, b in x2v
        idx += 1
        grid[idx, :] = [a, b]
        infos[idx] = info_example2(a, b)
    end

    xi0_pts = [(-1.0, 0.2), (0.0, 0.5), (1.0, 0.8), (0.5, 0.5)]
    xi0_w   = [0.25, 0.25, 0.25, 0.25]
    info0   = zeros(5, 5)
    for (w, (a, b)) in zip(xi0_w, xi0_pts)
        info0 .+= w .* info_example2(a, b)
    end

    solver = OWEASolver(grid, infos, Matrix(1.0I, 5, 5);
                        p=1, n0=40.0, n1=120.0, info_xi0=info0, eps_opt=1e-6)
    res = solve!(solver)
    @printf("A-optimal multistage objective Tr(Σ) at N=20²: %.6f (paper ≈ 0.203818)\n",
            res.objective_tilde)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Example 4 – model (10):  Y ~ θ₁e^{θ₂x} + θ₃e^{θ₄x}                      #
# ─────────────────────────────────────────────────────────────────────────── #

function info_example4_model10(x::Float64;
                               θ1=1.0, θ2=0.5, θ3=1.0, θ4=1.0)
    f = [exp(θ2 * x),
         θ1 * x * exp(θ2 * x),
         exp(θ4 * x),
         θ3 * x * exp(θ4 * x)]
    return f * f'
end

function run_example4_model10_check()
    println("\n=== Example 4: model (10) design check ===")
    n = 10000
    grid = collect(range(0.0, 1.0, length=n+1))
    infos = [info_example4_model10(x) for x in grid]
    grid_mat = reshape(grid, :, 1)

    # g(θ) = dη/dx|_{x=0} = θ₁θ₂ + θ₃θ₄  ⟹ ∂g/∂θ = [θ₂, θ₁, θ₄, θ₃]
    g = reshape([0.5, 1.0, 1.0, 1.0], 1, 4)

    solver = OWEASolver(grid_mat, infos, g; p=0, eps_opt=1e-6)
    res = solve!(solver)

    pts = vec(res.support_points)
    wts = res.weights
    order = sortperm(pts)
    pts = pts[order]
    wts = wts[order]

    # Merge adjacent support points
    merged_pts = Float64[]
    merged_wts = Float64[]
    tol = 2.5e-4
    for (x, w) in zip(pts, wts)
        if isempty(merged_pts) || abs(x - merged_pts[end]) > tol
            push!(merged_pts, x)
            push!(merged_wts, w)
        else
            nw = merged_wts[end] + w
            if nw > 0
                merged_pts[end] = (merged_pts[end] * merged_wts[end] + x * w) / nw
            end
            merged_wts[end] = nw
        end
    end

    @printf("time=%.3fs, iter=%d, max d=%.3e\n",
            res.elapsed_seconds, res.iterations,
            res.max_directional_derivative)
    println("Computed support and weights (rounded to 4 d.p.):")
    for (x, w) in zip(merged_pts, merged_wts)
        @printf("  x=%.4f, w=%.4f\n", x, w)
    end
    println("Paper target: (0,0.3509), (0.3011,0.4438), (0.7926,0.1491), (1,0.0562)")
end

function run_example4_runtime_compare()
    println("\n=== Example 4 model (10): runtime compare ===")
    t_paper = 0.42
    n = 10000
    grid = collect(range(0.0, 1.0, length=n+1))
    infos = [info_example4_model10(x) for x in grid]
    grid_mat = reshape(grid, :, 1)

    g = reshape([0.5, 1.0, 1.0, 1.0], 1, 4)
    solver = OWEASolver(grid_mat, infos, g;
                        p=0, eps_opt=1e-6, max_outer_iter=250)
    res = solve!(solver)
    @printf("ours: %.3fs, paper: %.2fs, ratio ours/paper: %.2f\n",
            res.elapsed_seconds, t_paper, res.elapsed_seconds / t_paper)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Logistic model example                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

function build_logistic_grid()
    levels = [-1.0, 1.0]
    vols   = range(25.0, 45.0, length=20001)  # step = 0.001

    # 16 binary combos × 20001 vol levels = 320016 rows
    combos = vec([(a, b, c, d)
                  for a in levels, b in levels, c in levels, d in levels])
    N = length(combos) * length(vols)
    grid = Matrix{Float64}(undef, N, 5)
    idx = 0
    for (a, b, c, d) in combos
        for v in vols
            idx += 1
            grid[idx, :] = [a, b, c, d, v]
        end
    end
    return grid
end

function build_info_logistic(grid::Matrix{Float64}, θ::Vector{Float64})
    N = size(grid, 1)
    k = length(θ)   # 7
    infos = Vector{Matrix{Float64}}(undef, N)

    @inbounds for n in 1:N
        # predictor = [1, A, B, ESD, Pulse, vol, ESD*Pulse]
        out = [1.0, grid[n,1], grid[n,2], grid[n,3], grid[n,4], grid[n,5],
               grid[n,3] * grid[n,4]]
        xβ = dot(out, θ)
        xβc = clamp(xβ, -500.0, 500.0)
        scale = exp(xβc / 2.0) / (1.0 + exp(xβc))
        f = scale .* out
        infos[n] = f * f'
    end
    return infos
end

function run_example_logistic()
    println("\n=== Logistic model (new example) ===")
    θ = [-7.5, 1.50, -0.2, -0.15, 0.25, 0.35, 0.4]

    t_grid = time()
    grid = build_logistic_grid()
    @printf("Grid: %d points  (%.3fs to build)\n", size(grid, 1), time() - t_grid)

    t_info = time()
    infos = build_info_logistic(grid, θ)
    @printf("Info matrices: computed in %.3fs\n", time() - t_info)

    solver = OWEASolver(grid, infos, Matrix(1.0I, 7, 7);
                        p=0, eps_opt=1e-6, max_outer_iter=300)
    res = solve!(solver)

    max_d = res.max_directional_derivative
    d_eff_lb = max_d < 100 ? exp(-max_d / 7.0) : 0.0

    @printf("\nD-optimal design found in %.3fs, %d outer iterations\n",
            res.elapsed_seconds, res.iterations)
    @printf("Max directional derivative  = %.3e  (D-efficiency lower bound ≥ %.4f)\n",
            max_d, d_eff_lb)
    @printf("Objective  log|Σ|           = %.6f\n", res.objective_tilde)

    # Display merged support
    pts = res.support_points
    wts = res.weights
    order = sortperm(collect(eachrow(pts)),
                     by=r -> (r[1], r[2], r[3], r[4], r[5]))

    merged = Tuple{Vector{Float64}, Float64}[]
    for i in order
        p = pts[i, :]
        w = wts[i]
        if !isempty(merged) &&
           p[1:4] == merged[end][1][1:4] &&
           abs(p[5] - merged[end][1][5]) < 0.5
            prev_p, prev_w = merged[end]
            new_w = prev_w + w
            mp = copy(prev_p)
            mp[5] = (prev_p[5] * prev_w + p[5] * w) / new_w
            merged[end] = (mp, new_w)
        else
            push!(merged, (copy(p), w))
        end
    end

    @printf("Number of support points    = %d  (before merge: %d)\n",
            length(merged), length(wts))
    println("\n  A    B   ESD  Pulse    vol     weight")
    println("  " * "-"^44)
    for (p, w) in merged
        @printf("  %+.0f   %+.0f   %+.0f   %+.0f  %7.3f  %.4f\n",
                p[1], p[2], p[3], p[4], p[5], w)
    end
end

function run_example_logistic_aopt()
    println("\n=== Logistic model: A-optimal design (p=1) ===")
    θ = [-7.5, 1.50, -0.2, -0.15, 0.25, 0.35, 0.4]

    t_grid = time()
    grid = build_logistic_grid()
    @printf("Grid: %d points  (%.3fs to build)\n", size(grid, 1), time() - t_grid)

    t_info = time()
    infos = build_info_logistic(grid, θ)
    @printf("Info matrices: computed in %.3fs\n", time() - t_info)

    solver = OWEASolver(grid, infos, Matrix(1.0I, 7, 7);
                        p=1, eps_opt=1e-6, max_outer_iter=300)
    res = solve!(solver)

    @printf("\nA-optimal design found in %.3fs, %d outer iterations\n",
            res.elapsed_seconds, res.iterations)
    @printf("Max directional derivative = %.3e\n", res.max_directional_derivative)
    @printf("Objective  Tr(Σ)           = %.6f\n", res.objective_tilde)
    @printf("Number of support points: %d\n", length(res.weights))

    pts = res.support_points
    wts = res.weights
    order = sortperm(collect(eachrow(pts)),
                     by=r -> (r[1], r[2], r[3], r[4], r[5]))

    merged = Tuple{Vector{Float64}, Float64}[]
    for i in order
        p = pts[i, :]
        w = wts[i]
        if !isempty(merged) &&
           p[1:4] == merged[end][1][1:4] &&
           abs(p[5] - merged[end][1][5]) < 0.5
            prev_p, prev_w = merged[end]
            new_w = prev_w + w
            mp = copy(prev_p)
            mp[5] = (prev_p[5] * prev_w + p[5] * w) / new_w
            merged[end] = (mp, new_w)
        else
            push!(merged, (copy(p), w))
        end
    end

    @printf("Support points after merging: %d  (before merge: %d)\n",
            length(merged), length(wts))
    println("\n  A    B   ESD  Pulse    vol     weight")
    println("  " * "-"^44)
    for (p, w) in merged
        @printf("  %+.0f   %+.0f   %+.0f   %+.0f  %7.3f  %.4f\n",
                p[1], p[2], p[3], p[4], p[5], w)
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

function main()
    run_example1_runtime_table()
    run_example1_checks()
    run_example2_check()
    run_example4_model10_check()
    run_example4_runtime_compare()
    run_example_logistic()
    run_example_logistic_aopt()
end

main()
