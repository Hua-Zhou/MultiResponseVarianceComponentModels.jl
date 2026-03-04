"""
Stochastic Lanczos Quadrature (SLQ) for log-determinant estimation.

Estimates `logdet(A) = tr(log(A))` using randomized probing and Lanczos iteration,
requiring only matrix-vector products with `A`. Compatible with any operator that
implements `mul!(dest::AbstractVector, A, src::AbstractVector)`.

Reference: Ubaru, Chen, Saad (2017) "Fast Estimation of tr(f(A)) via Stochastic Lanczos
Quadrature", SIAM J. Matrix Anal. Appl.
"""

"""
    SLQWorkspace{T}

Pre-allocated workspace for one Lanczos run of dimension `n` with `k` Krylov steps.

Fields:
- `Q`: `n × (k+1)` matrix of Lanczos basis vectors
- `α`: diagonal of the `k×k` tridiagonal `T` (length `k`)
- `β`: off-diagonal of `T` (length `k-1`)
- `tmp`: `n`-length buffer for matrix-vector products
"""
struct SLQWorkspace{T}
    Q::Matrix{T}
    α::Vector{T}
    β::Vector{T}
    tmp::Vector{T}
end

function SLQWorkspace(::Type{T}, n::Int, k::Int) where {T}
    return SLQWorkspace{T}(
        Matrix{T}(undef, n, k + 1),
        Vector{T}(undef, k),
        Vector{T}(undef, k - 1),
        Vector{T}(undef, n),
    )
end

"""
    lanczos!(ws::SLQWorkspace{T}, A, z::AbstractVector{T}) -> kdim::Int

Run `k = length(ws.α)` steps of the Lanczos algorithm on operator `A` starting from
probe vector `z`. Fills `ws.α` and `ws.β` with the diagonal and off-diagonal of the
resulting symmetric tridiagonal matrix `T`, and stores the Lanczos basis in `ws.Q`.

Returns the actual Krylov dimension `kdim ≤ k` (may be less than `k` on lucky breakdown).

The first Lanczos vector is `q₁ = z / ‖z‖`. The matvec `A * qⱼ` is computed via
`mul!(ws.tmp, A, qⱼ)`.
"""
function lanczos!(ws::SLQWorkspace{T}, A, z::AbstractVector{T}) where {T}
    k = length(ws.α)
    Q = ws.Q
    α = ws.α
    β = ws.β
    tmp = ws.tmp

    # q₁ = z / ‖z‖
    nrm = norm(z)
    q1 = view(Q, :, 1)
    @. q1 = z / nrm

    kdim = k
    for j in 1:k
        qj = view(Q, :, j)
        # tmp = A qⱼ
        mul!(tmp, A, qj)
        # αⱼ = qⱼᵀ tmp
        α[j] = dot(qj, tmp)
        # Orthogonalize: tmp -= αⱼ qⱼ + βⱼ₋₁ qⱼ₋₁
        @. tmp -= α[j] * qj
        if j > 1
            @. tmp -= β[j - 1] * view(Q, :, j - 1)
        end
        # βⱼ = ‖tmp‖ (off-diagonal entry)
        if j < k
            βj = norm(tmp)
            # Lucky breakdown: Krylov subspace is A-invariant
            if βj < sqrt(eps(T)) * abs(α[j])
                kdim = j
                break
            end
            β[j] = βj
            # qⱼ₊₁ = tmp / βⱼ
            @. Q[:, j + 1] = tmp / βj
        end
    end
    return kdim
end

"""
    gauss_logdet(α::AbstractVector{T}, β::AbstractVector{T}, kdim::Int) -> T

Compute `e₁ᵀ log(T) e₁` where `T` is the `kdim×kdim` symmetric tridiagonal matrix
with diagonal `α[1:kdim]` and off-diagonal `β[1:kdim-1]`.

This is the Gauss quadrature approximation to `qᵀ log(A) q` (for unit probe `q`),
evaluated by eigendecomposing `T`:

    e₁ᵀ log(T) e₁ = Σⱼ U[1,j]² · log(λⱼ)

where `T = U Λ Uᵀ`. Eigenvalues are clamped to `eps(T)` before taking the log to
guard against floating-point noise producing small negative values.
"""
function gauss_logdet(α::AbstractVector{T}, β::AbstractVector{T}, kdim::Int) where {T}
    Tsym = SymTridiagonal(α[1:kdim], β[1:min(kdim - 1, length(β))])
    F = eigen(Tsym)
    result = zero(T)
    @inbounds for j in 1:kdim
        result += F.vectors[1, j]^2 * log(max(F.values[j], eps(T)))
    end
    return result
end

"""
    slq_logdet(A, n::Int; nprobes=20, krylov_dim=20,
               probe_type=:gaussian, rng=Random.default_rng()) -> T

Estimate `logdet(A)` using Stochastic Lanczos Quadrature.

Uses the identity `logdet(A) = tr(log(A))` and the stochastic trace estimator:

    tr(log(A)) ≈ (1/nprobes) Σᵢ ‖zᵢ‖² · (e₁ᵀ log(Tᵢ) e₁)

where `Tᵢ` is the `krylov_dim×krylov_dim` tridiagonal produced by Lanczos on `A`
starting from probe `zᵢ`, and `e₁ᵀ log(Tᵢ) e₁` is evaluated by Gauss quadrature.

# Arguments
- `A`: any operator supporting `mul!(dest::AbstractVector, A, src::AbstractVector)`,
  e.g., `AbstractMatrix`, `ΩMatrix`, `VarCompStructure`.
- `n`: dimension of the operator (length of the vector space).

# Keywords
- `nprobes`: number of random probe vectors (more = lower variance).
- `krylov_dim`: number of Lanczos steps per probe (more = higher accuracy per probe).
- `probe_type`: `:gaussian` (entries `~ N(0,1)`) or `:rademacher` (entries `~ ±1`).
- `rng`: random number generator.

# Returns
Scalar estimate of `logdet(A)`.
"""
function slq_logdet(
    A,
    n::Int;
    nprobes::Int=20,
    krylov_dim::Int=20,
    probe_type::Symbol=:gaussian,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    T = Float64
    ws = SLQWorkspace(T, n, krylov_dim)
    z = Vector{T}(undef, n)
    estimate = zero(T)

    for _ in 1:nprobes
        _sample_probe!(z, rng, probe_type)
        nrm2 = dot(z, z)
        kdim = lanczos!(ws, A, z)
        estimate += nrm2 * gauss_logdet(ws.α, ws.β, kdim)
    end
    return estimate / nprobes
end

# Typed variant for non-Float64 operators.
function slq_logdet(
    A,
    n::Int,
    ::Type{T};
    nprobes::Int=20,
    krylov_dim::Int=20,
    probe_type::Symbol=:gaussian,
    rng::Random.AbstractRNG=Random.default_rng(),
) where {T<:AbstractFloat}
    ws = SLQWorkspace(T, n, krylov_dim)
    z = Vector{T}(undef, n)
    estimate = zero(T)

    for _ in 1:nprobes
        _sample_probe!(z, rng, probe_type)
        nrm2 = dot(z, z)
        kdim = lanczos!(ws, A, z)
        estimate += nrm2 * gauss_logdet(ws.α, ws.β, kdim)
    end
    return estimate / nprobes
end

# Internal: fill z with probe vector entries.
function _sample_probe!(z::AbstractVector{T}, rng, probe_type::Symbol) where {T}
    if probe_type === :gaussian
        randn!(rng, z)
    elseif probe_type === :rademacher
        @inbounds for i in eachindex(z)
            z[i] = rand(rng, Bool) ? one(T) : -one(T)
        end
    else
        throw(ArgumentError("probe_type must be :gaussian or :rademacher, got :$probe_type"))
    end
    return z
end
