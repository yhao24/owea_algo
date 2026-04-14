"""
    OWEA – Optimal Weights Exchange Algorithm

Julia implementation of the algorithm described in:
  Yang, Biedermann & Tang — "On Optimal Designs for Nonlinear Models:
  A General and Efficient Algorithm"

The solver supports:
  • Φ_p-optimality for integer p ≥ 0  (D when p=0, A when p=1)
  • Full parameter vector or sub-vector / differentiable function g(θ)
  • Locally optimal and multistage designs

Optimizations over the original implementation:
  • Precomputed stacked info matrix (k²×N) for O(k²N) batch directional-
    derivative evaluation via a single BLAS gemv call (info_stack' * h).
  • Newton weight optimisation reformulated via B[i] = C * Δ[i] and
    F[i] = inv_T * Δ[i] (C = g * inv_total), eliminating redundant
    matrix-chain products inside the O(m²) Hessian loop.
  • tr(A*B) computed via trdot (element-wise double loop, avoids
    allocating the k×k product matrix).
  • σ^l matrix powers precomputed once per Newton step (not rebuilt
    per (i,j) pair) for the p>1 Hessian second-sum term.
  • Δ[i] = a1*(I_{x_i} - I_{x_m}) precomputed once outside the
    Newton inner loop (they change only when support changes).
  • p=1 fast path uses tr(Σ) directly and Q₁=C'C for the Hessian,
    avoiding eigenvalue decomposition.
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
    # Flat k²×N matrix: column n = vec(I_x[n]).  Replaces Vector{Matrix{Float64}}
    # to avoid 48 M heap allocations (8.5 GB) vs 6.2 GB for this contiguous store.
    # Access element n as  reshape(view(s.info_matrices, :, n), s.k, s.k).
    info_matrices::Matrix{Float64}
    # Float32 copy of info_matrices for the batch BLAS sgemv in
    # directional_derivatives!  (3.1 GB).  Float32 is sufficient for argmax.
    info_stack_f32::Matrix{Float32}
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
                        info_matrices::AbstractMatrix,   # k²×N flat matrix
                        g_jacobian::AbstractMatrix;
                        p::Int = 0,
                        n0::Real = 0.0, n1::Real = 1.0,
                        info_xi0::Union{Nothing, AbstractMatrix} = nothing,
                        eps_opt::Real = 1e-6,
                        eps_weight::Real = 1e-10,
                        max_outer_iter::Int = 300,
                        max_newton_iter::Int = 60)
        kk, N = size(info_matrices)
        k  = round(Int, sqrt(kk))
        k * k == kk || error("info_matrices must have k²  rows")
        v  = size(g_jacobian, 1)
        total = Float64(n0) + Float64(n1)
        total > 0 || error("n0 + n1 must be positive")
        I0  = info_xi0 === nothing ? zeros(k, k) : Matrix{Float64}(info_xi0)
        g   = Matrix{Float64}(g_jacobian)
        gp  = Matrix{Float64}(grid_points)
        ims = Matrix{Float64}(info_matrices)   # ensure Float64, contiguous
        stk = Matrix{Float32}(ims)             # Float32 copy for sgemv

        new(gp, ims, stk, g, collect(g'), N, k, v, p,
            Float64(n0), Float64(n1), I0,
            Float64(n0)/total, Float64(n1)/total,
            Float64(eps_opt), Float64(eps_weight),
            max_outer_iter, max_newton_iter)
    end
end

"""Return a view of the n-th info matrix (k×k) from the flat k²×N storage."""
@inline function _imat(s::OWEASolver, n::Int)
    reshape(view(s.info_matrices, :, n), s.k, s.k)
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

"""
tr(A*B) via element-wise summation — avoids allocating the product matrix.
Uses the identity tr(A*B) = Σᵢⱼ Aᵢⱼ Bⱼᵢ.
"""
@inline function trdot(A::AbstractMatrix, B::AbstractMatrix)
    s = 0.0
    @inbounds for j in axes(A,2), i in axes(A,1)
        s += A[i,j] * B[j,i]
    end
    s
end

"""Invert a symmetric positive-(semi)definite matrix robustly."""
function spd_inv(A::AbstractMatrix)
    S = copy(A); sym!(S)
    n = size(S, 1)
    Id = Matrix{Float64}(I, n, n)
    try
        F = cholesky(Symmetric(S))
        return F \ Id
    catch end
    reg = 1e-14 * max(1.0, tr(S) / n)
    for _ in 1:8
        try
            F = cholesky(Symmetric(S .+ reg .* Id))
            return F \ Id
        catch
            reg *= 10.0
        end
    end
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
    if s.p == 1
        return tr(sigma)
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
        wj = w[j]
        col = view(s.info_matrices, :, idx)
        k = s.k
        for c in 1:k, r in 1:k
            out[r,c] += wj * col[(c-1)*k + r]
        end
    end
    sym!(out)
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Newton: objective, gradient, Hessian for weight vector u                  #
#                                                                             #
#  Key reformulation (avoids O(m²k³) matrix chains inside the loop):         #
#   C   = g * inv_total              (v×k)                                   #
#   B[i]= C * Δ[i]                  (v×k, Δ passed in by caller)             #
#   F[i]= inv_T * Δ[i]              (k×k)                                    #
#   dΣ/dω_i = -(B[i] * C')     — symmetric v×v                              #
#   d²Σ/dω_i dω_j = C*(Δ[j]*F[i]+Δ[i]*F[j])*C'   (from A.16)               #
#                                                                             #
#  tr-based contractions replace full product materialisation everywhere.    #
#  Δ[i] = a1*(I_{x_i}-I_{x_m}) is precomputed outside the Newton loop by    #
#  optimize_weights! and passed in so it is not recomputed each iteration.   #
# ─────────────────────────────────────────────────────────────────────────── #

function weight_eval_newton(s::OWEASolver, support::AbstractVector{Int},
                            u::AbstractVector{Float64},
                            Δ::Vector{Matrix{Float64}})
    m  = length(support)
    m1 = m - 1
    m1 <= 0 && return (0.0, Float64[], zeros(0,0))
    any(u .<= 0.0) && return nothing
    su = sum(u); su >= 1.0 && return nothing
    wm = 1.0 - su

    k = s.k
    # Build I_ξ from current weights using flat info_matrices columns
    info_xi = zeros(k, k)
    @inbounds for j in 1:m1
        col = view(s.info_matrices, :, support[j])
        for c in 1:k, r in 1:k; info_xi[r,c] += u[j] * col[(c-1)*k + r]; end
    end
    @inbounds begin
        col_m = view(s.info_matrices, :, support[m])
        for c in 1:k, r in 1:k; info_xi[r,c] += wm * col_m[(c-1)*k + r]; end
    end
    sym!(info_xi)

    info_total = s.a0 .* s.info_xi0 .+ s.a1 .* info_xi
    sym!(info_total)

    inv_T  = try spd_inv(info_total) catch; return nothing end
    C      = s.g * inv_T          # v×k
    sigma  = C * s.gT; sym!(sigma)

    # B[i] = C * Δ[i]  (v×k);  F[i] = inv_T * Δ[i]  (k×k)
    B = [C * Δ[i] for i in 1:m1]
    F = [inv_T * Δ[i] for i in 1:m1]

    grad = zeros(m1)
    hess = zeros(m1, m1)

    if s.p == 0
        ld = logdet(Symmetric(sigma))
        isfinite(ld) || return nothing
        objective = ld
        inv_sigma = try spd_inv(sigma) catch; return nothing end

        # WC = C' * Σ⁻¹  (k×v)
        WC = C' * inv_sigma   # k×v
        # Q = C' * Σ⁻¹ * C  (k×k) — for Hessian first term
        Q  = WC * C           # k×k

        # grad[i] = tr(Σ⁻¹ dΣ/dωᵢ) = -tr(Σ⁻¹ B[i] C')
        #         = -tr(C' Σ⁻¹ B[i])    (cyclic)
        #         = -trdot(WC, B[i])     [WC k×v, B[i] v×k → tr(WC*B[i])]
        @inbounds for i in 1:m1
            grad[i] = -trdot(WC, B[i])
        end

        # P[i] = WC * B[i] = C' * Σ⁻¹ * B[i]  (k×k)
        P = [WC * B[i] for i in 1:m1]

        # Hessian (A.18):
        #   H[i,j] = tr(Σ⁻¹ d²Σ/dωᵢdωⱼ) − tr(Σ⁻¹ dΣ/dωⱼ Σ⁻¹ dΣ/dωᵢ)
        #   First term:  tr(Q*(Δ[j]F[i]+Δ[i]F[j]))
        #              = trdot(Q*Δ[j],F[i]) + trdot(Q*Δ[i],F[j])
        #   Second term: tr(P[j]*P[i]) = trdot(P[j],P[i])
        @inbounds for j in 1:m1, i in 1:j
            t1 = trdot(Q * Δ[j], F[i]) + trdot(Q * Δ[i], F[j])
            t2 = trdot(P[j], P[i])
            hess[i,j] = t1 - t2
            hess[j,i] = hess[i,j]
        end

    elseif s.p == 1
        objective = tr(sigma)

        # grad[i] = tr(dΣ/dωᵢ) = -tr(B[i]*C') = -trdot(C', B[i])
        @inbounds for i in 1:m1
            grad[i] = -trdot(C', B[i])
        end

        # Hessian (A.20 with p=1, second sum vanishes):
        #   H[i,j] = tr(d²Σ/dωᵢdωⱼ) = tr(C(Δ[j]F[i]+Δ[i]F[j])C')
        #          = tr(C'C(Δ[j]F[i]+Δ[i]F[j]))
        Q1 = C' * C   # k×k
        @inbounds for j in 1:m1, i in 1:j
            t1 = trdot(Q1 * Δ[j], F[i]) + trdot(Q1 * Δ[i], F[j])
            hess[i,j] = t1
            hess[j,i] = t1
        end

    else
        # General p > 1  — Appendix A.19–A.20
        objective = tilde_phi(s, sigma)

        # Precompute σ^0 .. σ^{p-1} once (avoids repeated matrix powers per (i,j))
        pows = Vector{Matrix{Float64}}(undef, s.p)
        pows[1] = Matrix{Float64}(I, s.v, s.v)
        for l in 2:s.p
            pows[l] = pows[l-1] * sigma
        end
        sigma_pm1 = pows[s.p]   # σ^{p-1}

        # grad[i] = p * tr(σ^{p-1} dΣ/dωᵢ) = -p * tr(σ^{p-1} B[i] C')
        #         = -p * trdot(C * σ^{p-1}, B[i]')   — note C*(σ^{p-1})=(σ^{p-1}'*C')'
        # More simply: -p * trdot(B[i], (C * sigma_pm1)')
        #  = -p * sum_{a,b} B[i][a,b] * (sigma_pm1' * C')[b,a]
        #  = -p * tr(B[i] * sigma_pm1' * C') ... let's use:
        #    tr(σ^{p-1} B[i] C') = tr((C')' (σ^{p-1} B[i])') = trdot(C, sigma_pm1 * B[i])... Nope.
        # Direct: tr(σ^{p-1} B[i] C') = tr(C' σ^{p-1} B[i]) — cyclic
        # = trdot(C' * sigma_pm1, B[i])  where trdot(A,B)=Σᵢⱼ Aᵢⱼ Bⱼᵢ
        # grad[i] = -p * tr(C' σ^{p-1} B[i])   (cyclic of -p*tr(σ^{p-1} B[i] C'))
        #         = -p * trdot(Qp, B[i])          [Qp k×v, B[i] v×k]
        Qp = C' * sigma_pm1   # k×v
        @inbounds for i in 1:m1
            grad[i] = -s.p * trdot(Qp, B[i])
        end

        # Hessian first term:
        # p * tr(C'σ^{p-1}C * (Δ[j]F[i]+Δ[i]F[j]))
        Qp_full = Qp * C   # k×k = C' σ^{p-1} C

        # Hessian second term (A.20) corrected:
        # p * Σ_l tr(σ^l B[j] C' σ^{p-2-l} B[i] C')
        # = p * Σ_l tr((C'σ^l B[j]) * (C'σ^{p-2-l}B[i]))   (cyclic)
        # = p * Σ_l trdot(R[l+1][j], R[p_minus_2-l+1][i])  (no transpose)
        p_minus_2 = s.p - 2
        R = [[C' * (pows[l+1] * B[i]) for i in 1:m1] for l in 0:p_minus_2]

        @inbounds for j in 1:m1, i in 1:j
            t1 = s.p * (trdot(Qp_full * Δ[j], F[i]) + trdot(Qp_full * Δ[i], F[j]))
            t2 = 0.0
            for l in 0:p_minus_2
                t2 += trdot(R[l+1][j], R[p_minus_2-l+1][i])
            end
            hess[i,j] = t1 + s.p * t2
            hess[j,i] = hess[i,j]
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

        # Precompute Δ[i] = a1*(I_{x_i} - I_{x_m}) once (outside Newton loop).
        # They only change when the support changes.
        Im = Matrix(_imat(s, support[m]))
        Δ  = [s.a1 .* (Matrix(_imat(s, support[i])) .- Im) for i in 1:m-1]

        converged = false
        for _ in 1:s.max_newton_iter
            cur = weight_eval_newton(s, support, u, Δ)
            cur === nothing && break
            obj, grad, H = cur
            gnorm = norm(grad)
            # Use relative gradient norm to handle objectives of varying scale
            gnorm < 1e-8 * (1.0 + abs(obj)) && (converged = true; break)

            # Solve  H * step = grad  with regularised Cholesky
            step = nothing
            reg  = 0.0
            m1   = length(u)
            for _ in 1:10
                Hreg = H .+ reg .* Matrix{Float64}(I, m1, m1)
                try
                    F_ch = cholesky(Symmetric(Hreg))
                    step  = F_ch \ grad
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

            # Back-tracking line search — tolerance scaled by objective
            ls_tol = 1e-10 * (1.0 + abs(obj))
            α = 1.0; accepted = false
            while α >= 1e-5
                u_new = u .- α .* step
                if any(u_new .<= 0.0) || sum(u_new) >= 1.0
                    α *= 0.5; continue
                end
                nxt = weight_eval_newton(s, support, u_new, Δ)
                if nxt !== nothing && nxt[1] <= obj + ls_tol
                    u = u_new; accepted = true; break
                end
                α *= 0.5
            end
            # If line search fails but gradient is essentially zero relative to scale,
            # treat as converged to avoid spurious point removal
            if !accepted
                gnorm < 1e-5 * (1.0 + abs(obj)) && (converged = true)
                break
            end
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
        # Δ is automatically recomputed at top of next while iteration
    end
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Directional derivatives (batch BLAS-accelerated)                          #
#                                                                             #
#  d_p(x,ξ) = a1 * (tr(H_mat*I_x) - tr(H_mat*I_ξ))   for all x in grid     #
#  where H_mat = C' * W * C  (k×k),  C = g * inv_total                      #
#                                                                             #
#  All N tr values computed in one shot:                                      #
#    h = vec(H_mat)  (k²-vector, column-major)                               #
#    all_traces = info_stack' * h    (N-vector, one BLAS gemv)               #
# ─────────────────────────────────────────────────────────────────────────── #

function directional_derivatives!(dvals::Vector{Float64}, s::OWEASolver,
                                  info_xi::AbstractMatrix,
                                  info_total::AbstractMatrix)
    inv_total = spd_inv(info_total)
    C     = s.g * inv_total    # v×k
    sigma = C * s.gT; sym!(sigma)

    if s.p == 0
        inv_sigma = spd_inv(sigma)
        W = inv_sigma
    else
        sigma_S   = Symmetric(sigma)
        sigma_p   = Matrix(sigma_S^s.p)
        tr_sp     = tr(sigma_p)
        scalar    = (1.0 / s.v)^(1.0 / s.p) * tr_sp^(1.0 / s.p - 1.0)
        sigma_pm1 = Matrix(sigma_S^(s.p - 1))
        W = scalar .* sigma_pm1
    end

    H_mat = C' * W * C    # k×k
    h     = vec(H_mat)     # k²-vector (column-major, matches info_matrices columns)

    # BLAS sgemv using Float32 for 2× throughput on large grids.
    h32        = Float32.(h)
    all_traces = similar(h32, s.N)       # preallocate Float32 result
    mul!(all_traces, transpose(s.info_stack_f32), h32)   # in-place sgemv, no alloc

    base = dot(h, vec(info_xi))      # tr(H_mat * I_ξ)

    a1 = s.a1
    base32 = Float32(base)
    @inbounds for n in 1:s.N
        dvals[n] = a1 * Float64(all_traces[n] - base32)
    end
    dvals
end

# ─────────────────────────────────────────────────────────────────────────── #
#  Greedy Fedorov-style initialisation                                        #
# ─────────────────────────────────────────────────────────────────────────── #

function greedy_init(s::OWEASolver)
    N = s.N; k = s.k; cnt = min(k+1, N)
    pool_sz = min(N, max(20k, 2000)); step = max(1, N÷pool_sz)
    pool = collect(1:step:N); length(pool) > pool_sz && resize!(pool, pool_sz)

    best = pool[argmax([tr(_imat(s, i)) for i in pool])]
    sup  = [best]
    ic   = Matrix(_imat(s, best))
    ic .+= 1e-12 .* Matrix{Float64}(I, k, k)
    for _ in 2:cnt
        iv = spd_inv(ic); bg = -1.0; bi = -1
        for i in pool
            i ∈ sup && continue
            g_cand = trdot(iv, _imat(s, i))
            g_cand > bg && (bg = g_cand; bi = i)
        end
        bi < 0 && break   # no eligible candidate (pool exhausted)
        push!(sup, bi); ic .+= _imat(s, bi)
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

    # Deduplicate: if the same grid index appears more than once (edge case),
    # merge by summing the corresponding weights.
    if length(unique(support)) < length(support)
        seen   = Dict{Int,Int}()   # grid_index => position in merged arrays
        m_sup  = Int[]
        m_wts  = Float64[]
        for (i, idx) in enumerate(support)
            if haskey(seen, idx)
                m_wts[seen[idx]] += weights[i]
            else
                seen[idx] = length(m_sup) + 1
                push!(m_sup, idx); push!(m_wts, weights[i])
            end
        end
        sw = sum(m_wts); sw > 0 && (m_wts ./= sw)
        support = m_sup; weights = m_wts
    end

    OWEAResult(support, s.grid_points[support, :], weights,
               obj, best_max_d, it, time() - t0)
end

end # module OWEA
