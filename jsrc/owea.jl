"""
    OWEA – Optimal Weights Exchange Algorithm

Julia implementation of the algorithm described in:
  Yang, Biedermann & Tang — "On Optimal Designs for Nonlinear Models:
  A General and Efficient Algorithm"

The solver supports:
  • Φ_p-optimality for integer p ≥ 0  (D when p=0, A when p=1)
  • Full parameter vector or sub-vector / differentiable function g(θ)
  • Locally optimal and multistage designs
"""
module OWEA

using LinearAlgebra

export OWEAResult, OWEASolver, solve!

# ─────────────────────────────────────────────────────────────────────────── #
#  Data types                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

struct OWEAResult
    support_indices::Vector{Int}
    support_points::Matrix{Float64}
    weights::Vector{Float64}
    objective_tilde::Float64
    max_directional_derivative::Float64
    iterations::Int
    elapsed_seconds::Float64
end

struct OWEASolver
    grid_points::Matrix{Float64}
    info_matrices::Vector{Matrix{Float64}}
    g::Matrix{Float64}           # v × k
    gT::Matrix{Float64}          # k × v

    N::Int; k::Int; v::Int
    p::Int
    n0::Float64; n1::Float64
    info_xi0::Matrix{Float64}
    a0::Float64; a1::Float64

    eps_opt::Float64
    eps_weight::Float64
    max_outer_iter::Int
    max_newton_iter::Int

    function OWEASolver(grid_points::AbstractMatrix,
                        info_matrices::AbstractVector{<:AbstractMatrix},
                        g_jacobian::AbstractMatrix;
                        p::Int = 0,
                        n0::Real = 0.0, n1::Real = 1.0,
                        info_xi0::Union{Nothing, AbstractMatrix} = nothing,
                        eps_opt::Real = 1e-6,
                        eps_weight::Real = 1e-10,
                        max_outer_iter::Int = 300,
                        max_newton_iter::Int = 60)
        N = size(grid_points, 1)
        k = size(first(info_matrices), 1)
        v = size(g_jacobian, 1)
        total = Float64(n0) + Float64(n1)
        total > 0 || error("n0 + n1 must be positive")
        I0 = info_xi0 === nothing ? zeros(k, k) : Matrix{Float64}(info_xi0)
        g  = Matrix{Float64}(g_jacobian)
        gp = Matrix{Float64}(grid_points)
        ims = [Matrix{Float64}(m) for m in info_matrices]
        new(gp, ims, g, collect(g'), N, k, v, p,
            Float64(n0), Float64(n1), I0,
            Float64(n0)/total, Float64(n1)/total,
            Float64(eps_opt), Float64(eps_weight),
            max_outer_iter, max_newton_iter)
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Numerically stable helpers                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@inline function sym!(A::AbstractMatrix)
    @inbounds for j in axes(A, 2), i in 1:j-1
        v = (A[i,j] + A[j,i]) * 0.5
        A[i,j] = v; A[j,i] = v
    end
    A
end

"""Invert a symmetric positive-(semi)definite matrix robustly."""
function spd_inv(A::AbstractMatrix)
    S = copy(A); sym!(S)
    n = size(S, 1)
    Id = Matrix{Float64}(I, n, n)
    # Try Cholesky first (fast & accurate)
    try
        F = cholesky(Symmetric(S))
        return F \ Id
    catch
    end
    # Add small regularisation and retry
    reg = 1e-14 * max(1.0, tr(S) / n)
    for _ in 1:8
        try
            F = cholesky(Symmetric(S .+ reg .* Id))
            return F \ Id
        catch
            reg *= 10.0
        end
    end
    # Last resort: general LU/SVD
    return pinv(S)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Σ(g) and objective                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

@inline function combined_info!(out::Matrix{Float64}, s::OWEASolver,
                                info_xi::AbstractMatrix)
    @inbounds for j in axes(out, 2), i in axes(out, 1)
        out[i,j] = s.a0 * s.info_xi0[i,j] + s.a1 * info_xi[i,j]
    end
    sym!(out)
end

function sigma_mat(s::OWEASolver, info_total::AbstractMatrix)
    inv_t = spd_inv(info_total)
    M = s.g * inv_t * s.gT
    sym!(M)
end

function tilde_phi(s::OWEASolver, sigma::AbstractMatrix)
    if s.p == 0
        ld = logdet(Symmetric(sigma))
        return isfinite(ld) ? ld : Inf
    end
    ev = eigvals(Symmetric(sigma))
    clamp!(ev, 1e-18, Inf)
    return sum(x -> x^s.p, ev)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Design info from support + weights                                         #
# ─────────────────────────────────────────────────────────────────────────── #

function design_info!(out::Matrix{Float64}, s::OWEASolver,
                      support::AbstractVector{Int}, w::AbstractVector{Float64})
    fill!(out, 0.0)
    @inbounds for (j, idx) in enumerate(support)
        wj = w[j]; M = s.info_matrices[idx]
        for c in axes(M, 2), r in axes(M, 1)
            out[r,c] += wj * M[r,c]
        end
    end
    sym!(out)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Newton: objective, gradient, Hessian for weight vector u                    #
# ─────────────────────────────────────────────────────────────────────────── #

function weight_eval_newton(s::OWEASolver, support::AbstractVector{Int},
                            u::AbstractVector{Float64})
    m  = length(support)
    m1 = m - 1
    m1 <= 0 && return (0.0, Float64[], zeros(0,0))
    any(u .<= 0.0) && return nothing
    su = sum(u); su >= 1.0 && return nothing
    wm = 1.0 - su

    k = s.k
    info_xi = zeros(k, k)
    @inbounds for j in 1:m1
        M = s.info_matrices[support[j]]
        for c in 1:k, r in 1:k
            info_xi[r,c] += u[j] * M[r,c]
        end
    end
    @inbounds begin
        Mm = s.info_matrices[support[m]]
        for c in 1:k, r in 1:k
            info_xi[r,c] += wm * Mm[r,c]
        end
    end
    sym!(info_xi)

    info_total = s.a0 .* s.info_xi0 .+ s.a1 .* info_xi
    sym!(info_total)

    inv_total = try spd_inv(info_total) catch; return nothing end

    sigma = s.g * inv_total * s.gT
    sym!(sigma)

    Mm = s.info_matrices[support[m]]
    deltas = Vector{Matrix{Float64}}(undef, m1)
    @inbounds for i in 1:m1
        deltas[i] = s.a1 .* (s.info_matrices[support[i]] .- Mm)
    end

    ds = Vector{Matrix{Float64}}(undef, m1)
    for i in 1:m1
        tmp = -(s.g * (inv_total * (deltas[i] * (inv_total * s.gT))))
        sym!(tmp)
        ds[i] = tmp
    end

    grad = zeros(m1)
    hess = zeros(m1, m1)

    if s.p == 0
        ld = logdet(Symmetric(sigma))
        isfinite(ld) || return nothing
        objective = ld
        inv_sigma = try spd_inv(sigma) catch; return nothing end

        @inbounds for i in 1:m1
            grad[i] = tr(inv_sigma * ds[i])
        end
        @inbounds for j in 1:m1, i in 1:m1
            mid = deltas[j] * (inv_total * (deltas[i])) .+
                  deltas[i] * (inv_total * (deltas[j]))
            d2 = s.g * (inv_total * (mid * (inv_total * s.gT)))
            sym!(d2)
            hess[i,j] = tr(inv_sigma * d2) -
                         tr((inv_sigma * ds[j]) * (inv_sigma * ds[i]))
        end

    elseif s.p == 1
        objective = tr(sigma)
        @inbounds for i in 1:m1
            grad[i] = tr(ds[i])
        end
        @inbounds for j in 1:m1, i in 1:m1
            mid = deltas[j] * (inv_total * (deltas[i])) .+
                  deltas[i] * (inv_total * (deltas[j]))
            d2 = s.g * (inv_total * (mid * (inv_total * s.gT)))
            hess[i,j] = tr(d2)
        end

    else
        # General p > 1 (A.19)-(A.20)
        sigma_S = Symmetric(sigma)
        objective = tilde_phi(s, sigma)
        sigma_pm1 = Matrix(sigma_S^(s.p - 1))

        @inbounds for i in 1:m1
            grad[i] = s.p * tr(sigma_pm1 * ds[i])
        end
        @inbounds for j in 1:m1, i in 1:m1
            mid = deltas[j] * (inv_total * (deltas[i])) .+
                  deltas[i] * (inv_total * (deltas[j]))
            d2 = s.g * (inv_total * (mid * (inv_total * s.gT)))
            val = s.p * tr(sigma_pm1 * d2)
            for l in 0:(s.p-2)
                Sl  = l == 0         ? I : Matrix(sigma_S^l)
                Spl = s.p-2-l == 0  ? I : Matrix(sigma_S^(s.p-2-l))
                val += s.p * tr(Sl * ds[j] * Spl * ds[i])
            end
            hess[i,j] = val
        end
    end

    sym!(hess)
    return (objective, grad, hess)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Weight optimisation (Newton + boundary removal)                             #
# ─────────────────────────────────────────────────────────────────────────── #

function optimize_weights!(s::OWEASolver, support::Vector{Int},
                           w0::Vector{Float64})
    min_support = s.n0 == 0.0 ? s.k : 1
    w = copy(w0)
    if length(w) != length(support) || sum(w) <= 0.0
        w = fill(1.0 / length(support), length(support))
    else
        clamp!(w, 0.0, Inf); w ./= sum(w)
    end

    while true
        m = length(support)
        m == 1 && return (support, [1.0])
        clamp!(w, 1e-14, Inf); w ./= sum(w)
        u = w[1:end-1]

        converged = false
        for _ in 1:s.max_newton_iter
            cur = weight_eval_newton(s, support, u)
            cur === nothing && break
            obj, grad, H = cur
            gnorm = norm(grad)
            gnorm < 1e-9 && (converged = true; break)

            # Solve  H step = grad  with regularised Cholesky
            step = nothing
            reg = 0.0
            m1 = length(u)
            for _ in 1:10
                Hreg = H .+ reg .* Matrix{Float64}(I, m1, m1)
                try
                    F = cholesky(Symmetric(Hreg))
                    step = F \ grad
                    break
                catch
                    reg = reg == 0.0 ?
                          max(1e-12, 1e-10 * gnorm) :
                          reg * 10.0
                end
            end
            if step === nothing
                try step = pinv(H) * grad catch; break end
            end

            # Back-tracking line search
            α = 1.0; accepted = false
            while α >= 1e-5
                u_new = u .- α .* step
                if any(u_new .<= 0.0) || sum(u_new) >= 1.0
                    α *= 0.5; continue
                end
                nxt = weight_eval_newton(s, support, u_new)
                if nxt !== nothing && nxt[1] <= obj + 1e-14
                    u = u_new; accepted = true; break
                end
                α *= 0.5
            end
            accepted || break
        end

        w = vcat(u, 1.0 - sum(u))
        clamp!(w, 0.0, Inf)
        if converged && all(w .> s.eps_weight)
            w ./= sum(w); return (support, w)
        end

        length(support) <= min_support && begin
            clamp!(w, s.eps_weight, Inf); w ./= sum(w)
            return (support, w)
        end

        rm = argmin(w)
        deleteat!(support, rm); deleteat!(w, rm)
        sw = sum(w)
        sw <= 0.0 ? (w = fill(1.0/length(support), length(support))) :
                      (w ./= sw)
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Directional derivatives                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

function directional_derivatives!(dvals::Vector{Float64}, s::OWEASolver,
                                  info_xi::AbstractMatrix,
                                  info_total::AbstractMatrix)
    inv_total = spd_inv(info_total)
    sigma = s.g * inv_total * s.gT; sym!(sigma)
    C = s.g * inv_total

    if s.p == 0
        inv_sigma = spd_inv(sigma)
        W = inv_sigma
    else
        sigma_S = Symmetric(sigma)
        sigma_p  = Matrix(sigma_S^s.p)
        tr_sp    = tr(sigma_p)
        scalar   = (1.0 / s.v)^(1.0 / s.p) * tr_sp^(1.0 / s.p - 1.0)
        sigma_pm1 = Matrix(sigma_S^(s.p - 1))
        W = scalar .* sigma_pm1
    end

    H = C' * W * C      # k × k
    base = tr(H * info_xi)

    @inbounds for n in 1:s.N
        Ix = s.info_matrices[n]
        t = 0.0
        for c in 1:s.k, r in 1:s.k
            t += H[r,c] * Ix[r,c]
        end
        dvals[n] = s.a1 * (t - base)
    end
    dvals
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Greedy Fedorov-style initialisation                                        #
# ─────────────────────────────────────────────────────────────────────────── #

function greedy_init(s::OWEASolver)
    N = s.N; k = s.k; cnt = min(k+1, N)
    pool_sz = min(N, 20k); step = max(1, N÷pool_sz)
    pool = collect(1:step:N); length(pool) > pool_sz && resize!(pool, pool_sz)

    best = pool[argmax([tr(s.info_matrices[i]) for i in pool])]
    sup = [best]
    ic = copy(s.info_matrices[best])
    ic .+= 1e-12 .* Matrix{Float64}(I, k, k)
    for _ in 2:cnt
        iv = spd_inv(ic); bg = -1.0; bi = pool[1]
        for i in pool
            i ∈ sup && continue
            g = tr(iv * s.info_matrices[i])
            g > bg && (bg = g; bi = i)
        end
        push!(sup, bi); ic .+= s.info_matrices[bi]
    end
    (sup, fill(1.0/length(sup), length(sup)))
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Main loop                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

function solve!(s::OWEASolver)::OWEAResult
    t0 = time()

    init_count = min(s.k + 1, s.N)
    init_idx = unique(round.(Int, range(1, s.N, length=init_count)))
    support = copy(init_idx)
    weights = fill(1.0 / length(support), length(support))

    info_buf  = zeros(s.k, s.k)
    total_buf = zeros(s.k, s.k)
    design_info!(info_buf, s, support, weights)
    combined_info!(total_buf, s, info_buf)
    if rank(Symmetric(total_buf), atol=1e-10) < s.k
        support, weights = greedy_init(s)
    end

    dvals   = zeros(s.N)
    in_supp = falses(s.N)

    best_max_d = Inf; prev_best = Inf; no_improve = 0; it = 0

    while it < s.max_outer_iter
        it += 1
        support, weights = optimize_weights!(s, support, weights)

        keep_thr = max(s.eps_weight, 1e-7)
        keep = weights .> keep_thr
        if !all(keep)
            support = support[keep]; weights = weights[keep]
            weights ./= sum(weights)
        end

        design_info!(info_buf, s, support, weights)
        combined_info!(total_buf, s, info_buf)
        try
            directional_derivatives!(dvals, s, info_buf, total_buf)
        catch
            break
        end

        best_max_d = maximum(dvals)
        if prev_best - best_max_d <= 1e-10
            no_improve += 1
        else
            no_improve = 0
        end
        prev_best = best_max_d

        fill!(in_supp, false)
        for idx in support; in_supp[idx] = true end

        best_new_idx = 0; best_new_d = -Inf
        @inbounds for n in 1:s.N
            if !in_supp[n] && dvals[n] > best_new_d
                best_new_d = dvals[n]; best_new_idx = n
            end
        end

        best_max_d <= s.eps_opt && break
        no_improve >= 10         && break

        if best_new_idx > 0 && best_new_d > s.eps_opt
            push!(support, best_new_idx)
            push!(weights, 0.0)
        else
            break
        end
    end

    design_info!(info_buf, s, support, weights)
    combined_info!(total_buf, s, info_buf)
    obj = try
        sig = sigma_mat(s, total_buf)
        tilde_phi(s, sig)
    catch
        Inf
    end

    OWEAResult(support, s.grid_points[support, :], weights,
               obj, best_max_d, it, time() - t0)
end

end # module OWEA
