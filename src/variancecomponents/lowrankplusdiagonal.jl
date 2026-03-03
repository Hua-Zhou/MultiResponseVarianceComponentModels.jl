struct LowRankPlusDiagonal{T} <: VarCompStructure{T}
    # parameters
    F::Matrix{T}
    Ψ::Vector{T}
    # working arrays
    Σ::Matrix{T}
    # storage_nd_nd::Matrix{T}
    storage_nn::Matrix{T}
    storage_dd_1::Matrix{T}
    storage_dd_2::Matrix{T}
    storage_dd_3::Matrix{T}
    # Constants
    Σrank::Int
    V::Matrix{T}
    Vrank::Int
    pardim::Int
    # LAPACK workspace
    lworkspace::Vector{T}
    rworkspace::Vector{T}
    info::Ref{BlasInt}
end

## Constructors ----------------
function LowRankPlusDiagonal(F::Matrix{T}, Ψ::Vector{T}, V::Matrix{T}) where {T}
    n = checksquare(V)
    d, r = size(F)
    nd = n * d
    (length(Ψ) == d) || throw(DimensionMismatch())
    Σ = Matrix{T}(undef, d, d)
    BLAS.syrk!('L', 'N', one(T), F, zero(T), Σ)
    copytri!(Σ, 'L')
    @inbounds for i in 1:d
        Σ[i, i] += Ψ[i]
    end
    # dimension here is the number of elements in augmented matrix [F | Ψ]
    pardim = d * (r + 1)
    # Here, `Σrank` is not the rank of Σ = FFᵀ + Ψ, but the number of columns in F
    Σrank = r
    # storage_nd_nd = Matrix{T}(undef, nd, nd)
    storage_nn = Matrix{T}(undef, n, n)
    storage_dd_1 = Matrix{T}(undef, d, d)
    storage_dd_2 = Matrix{T}(undef, d, d)
    storage_dd_3 = Matrix{T}(undef, d, d)
    # TODO: assume V is full rank, in general V can be arbitrary rank
    Vrank = n
    # Pre-allocate LAPACK workspace needed for eigen decompositions
    eigdim = _syev_lwork!('V', 'L', Σ)
    lworkspace = Vector{T}(undef, eigdim)
    eigdim = _syev_lwork!('V', 'L', Matrix{T}(undef, r, r))
    rworkspace = Vector{T}(undef, eigdim)
    info = Ref{BlasInt}()
    return LowRankPlusDiagonal{T}(
        F,
        Ψ,
        Σ,
        # storage_nd_nd,
        storage_nn,
        storage_dd_1,
        storage_dd_2,
        storage_dd_3,
        Σrank,
        V,
        Vrank,
        pardim,
        lworkspace,
        rworkspace,
        info,
    )
end

# Constructor when only given dimensions of F along with V
function LowRankPlusDiagonal(d::Int, r::Int, V::Matrix{T}) where {T}
    F = Matrix{T}(undef, d, r)
    fill!(F, zero(T))
    dg = Vector{T}(one(T), d)
    return LowRankPlusDiagonal(F, dg, V)
end

# Constructor when given Σ, rank of F, and V
function LowRankPlusDiagonal(
    Σ::Matrix{T}, r::Int, V::Matrix{T}; eps::T=T(1e-8)
) where {T}
    # Best rank-r approximation to Σ
    d = checksquare(Σ)
    if isdiag(Σ) == true
        @warn "initial Σ is diagonal, returning F = 0 and Ψ = diag(Σ)"
        return LowRankPlusDiagonal(zeros(T, d, r), diag(Σ), V)
    end
    # by convention, `eigen` is normally smallest to largest
    # permute the eigenvalues and eigenvectors to largest to smallest
    Σ_eig = eigen(Σ; sortby=x -> -abs(x))
    F = Matrix{T}(undef, d, r)
    for j in 1:r
        sqrtλ = sqrt(Σ_eig.values[j])
        for i in 1:d
            F[i, j] = sqrtλ * Σ_eig.vectors[i, j]
        end
    end
    # Initialize dg with the residual, diag(Σ - FFᵀ)
    # ensure initial point is positive
    dg = diag(Σ) .- vec(sum(abs2, F; dims=2)) .+ eps
    return LowRankPlusDiagonal(F, dg, V)
end

# initialize F and Ψ with best low-rank plus diagonal approximation to S
function initialize!(
    VC::LowRankPlusDiagonal{T}, S::Matrix{T}; eps::T=T(1e-8)
) where {T}
    F = VC.F
    Ψ = VC.Ψ
    Σ = VC.Σ
    # Best diagonal plus rank-r approximation to S
    d = checksquare(S)
    r = VC.Σrank
    if isdiag(S) == true
        @warn "initial Σ is diagonal, returning F = 0 and Ψ = diag(Σ)"
        return LowRankPlusDiagonal(zeros(T, d, r), diag(S), VC.V)
    end
    # by convention, `eigen` is normally smallest to largest
    # permute the eigenvalues and eigenvectors to largest to smallest
    S_eig = eigen(S; sortby=x -> -abs(x))
    for j in 1:r
        sqrtλ = sqrt(S_eig.values[j])
        for i in 1:d
            F[i, j] = sqrtλ * S_eig.vectors[i, j]
        end
    end
    # Initialize Ψ with the residual, diag(S - FFᵀ)
    # ensure initial point is positive
    Ψ .= diag(S) .- vec(sum(abs2, F; dims=2)) .+ eps
    mul!(Σ, F, transpose(F), one(T), zero(T))
    for j in 1:d
        Σ[j, j] += Ψ[j]
    end
    return nothing
end

function initialize!(VC::LowRankPlusDiagonal{T}; eps::T=T(1e-8)) where {T}
    return initialize!(VC, VC.Σ; eps=eps)
end

# Update working estimate of Σ in place, without modifying F or Ψ
function materialize!(VC::LowRankPlusDiagonal{T}) where {T}
    F = VC.F
    Ψ = VC.Ψ
    Σ = VC.Σ
    BLAS.syrk!('L', 'N', one(T), F, zero(T), Σ)
    @inbounds for i in eachindex(Ψ)
        Σ[i, i] += Ψ[i]
    end
    return copytri!(Σ, 'L')
end

@inline function update_Ψ!(VC::LowRankPlusDiagonal{T}; eps::T=T(1e-8)) where {T}
    Ψ = VC.Ψ
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    # storage_d = view(VC.storage_dd_3, :, 1)
    # fill!(storage_d, zero(T))
    # for j in axes(VC.F, 2)
    #     storage_d .+= abs2.(view(VC.F, :, j))
    # end

    # storage for candidate update to Ψ, to avoid modifying Ψ 
    # until we check the rankedness condition
    Ψnew = view(VC.storage_nn, eachindex(Ψ))
    # Ψ must have at least d - r elements greater than zero
    # more stable to have constant diagonal === PCA-like model
    rank_lower_bound = length(Ψ) - VC.Σrank
    # the MM update of Ψ is a multiplicative scaling
    # this may lead to continued shrinking of Ψ without any escape
    rankedness = length(Ψ)
    @inbounds for i in eachindex(Ψ)
        Ψnew[i] = sqrt(N[i, i] / M[i, i]) * Ψ[i]
        if Ψnew[i] < eps
            rankedness -= 1
            Ψnew[i] = eps
        end
    end
    if rankedness < rank_lower_bound
        @debug "Ψ update has fewer than $(rank_lower_bound) elements greater than $(eps), skipping update for Ψ"
    else
        copyto!(Ψ, Ψnew)
    end
    return Ψ
end

@inline function _update_W!(
    W::AbstractMatrix{T}, H::AbstractMatrix{T}, F::AbstractMatrix{T}
) where {T}
    # update W in-place, W = HᵀH + Iᵣ ≡ FᵀΨ⁻¹F + Iᵣ
    BLAS.syrk!('L', 'T', one(T), H, zero(T), W)
    @inbounds for i in axes(F, 2)
        W[i, i] += one(T)
    end
    return copytri!(W, 'L')
end

@inline function _update_H!(
    H::AbstractMatrix{T}, F::AbstractMatrix{T}, sqrtΨ::AbstractVector{T}
) where {T}
    # update H in-place, H = (√Ψ) \ F
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        H[i, j] = F[i, j] / sqrtΨ[i]
    end
    return H
end

function update_F!(VC::LowRankPlusDiagonal{T}) where {T}
    d, r = size(VC.F)
    F = VC.F
    Ψ = VC.Ψ
    Σ = VC.Σ
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    C = VC.storage_dd_3
    info = VC.info
    storage_dd = view(VC.storage_nn, 1:d, 1:d)
    # avoid same memory space as d by d storage 
    sqrtΨ = copyto!(view(VC.storage_nn, 1:d, (d + 1)), Ψ)
    sqrtΨ .= sqrt.(Ψ)
    # assume current estimate of Σ(n) is stored in `VC.Σ`
    # C(n) = Σ(n) N(n) Σ(n)
    _update_XYX!(C, N, Σ, storage_dd)
    # Intermediate array M̃ = √Ψ M(n) √Ψ
    M_tilde = view(VC.storage_nn, (d + 1):(2d), 1:d)
    for j in axes(C, 2), i in axes(C, 1)
        M_tilde[i, j] = M[i, j] * sqrtΨ[j] * sqrtΨ[i]
    end
    # Intermediate array C̃ = (√Ψ)⁻¹ C(n) (√Ψ)⁻¹
    C_tilde = view(VC.storage_nn, (2d + 1):(3d), 1:d)
    for j in axes(C, 2), i in axes(C, 1)
        C_tilde[i, j] = C[i, j] / (sqrtΨ[j] * sqrtΨ[i])
    end
    # intermediate array H = (√Ψ) \ F and W = HᵀH + Iᵣ = FᵀΨ⁻¹F + Iᵣ
    H = view(VC.storage_nn, (3d + 1):(4d), 1:r)
    _update_H!(H, F, sqrtΨ)
    # solution to Riccati equation S M̃ S = C̃
    # S* = √M̃ √(√M̃ C̃ √M̃) √M̃
    # solve by eigendecomposition
    # Matrix square root of M̃
    sqrt_M_tilde = view(VC.storage_nn, (4d + 1):(5d), 1:d)
    λ_M_tilde    = view(VC.storage_nn, 1:d, d + 2)
    sqrt!(sqrt_M_tilde, M_tilde, λ_M_tilde, storage_dd, VC.lworkspace, info)
    Q_M_tilde = M_tilde
    # inner term of solution to Riccati equation, √(√M̃ C̃ √M̃)
    _update_XYX!(C, C_tilde, sqrt_M_tilde, storage_dd)
    λ_kernel = view(VC.storage_nn, 1:d, d + 3)
    kernel = sqrt!(C_tilde, C, λ_kernel, storage_dd, VC.lworkspace, info)
    # solution to Riccati equation
    # can overwrite memory of C
    inv_sqrt_M_tilde = sqrt!(
        sqrt_M_tilde, Q_M_tilde, λ_M_tilde, storage_dd; invsqrt=true
    )
    S = _update_XYX!(C, kernel, inv_sqrt_M_tilde, storage_dd)
    _syev!('V', 'L', S, λ_kernel, VC.lworkspace, info)
    info[] > 0 && throw(PosDefException(info[]))
    # by convention, `eigen` is normally smallest to largest
    # permute the eigenvalues and eigenvectors to largest to smallest
    # could write for loop from d:-1:d-r+1 to avoid the reverse, but this is more explicit
    reverse!(λ_kernel; dims=1)
    reverse!(S; dims=2)
    any(λ_kernel .< 1) &&
        @debug "Riccati solution has eigenvalues < 1, no (real) solution to surrogate objective exists."
    for j in 1:r
        λj = λ_kernel[j] > 1 ? λ_kernel[j] - 1 : zero(T)
        view(H, :, j) .= sqrt(λj) .* view(S, :, j)
    end
    # Sylvester based update (questionable)----------------------------
    # H_Winv = view(VC.storage_nn, (3d + 1):(4d), (r + 1):(2r))
    # W      = view(VC.storage_nn, (4d + 1):(4d + r), 1:r)
    # # Factorize W since we need W⁻¹
    # _, info[] = LAPACK.potrf!('L', W)
    # info[] > 0 && throw(PosDefException(info[]))
    # BLAS.trsm!('R', 'L', 'T', 'N', one(T), W, copyto!(H_Winv, H))
    # BLAS.trsm!('R', 'L', 'N', 'N', one(T), W, H_Winv)
    # # space allocated to W can now be re-used
    # # RHS of update equation
    # RHS = view(VC.storage_nn, (3d + 1):(4d), (2r + 1):(3r))
    # #  RHS = C̃ (H W⁻¹)
    # BLAS.symm!('L', 'L', one(T), C_tilde, H_Winv, zero(T), RHS)
    # Recycle space allocated to W
    # Winv_HtC̃H_Winv = W
    # (H W⁻¹)ᵀ C̃ (H W⁻¹)
    # mul!(Winv_HtC̃H_Winv, transpose(H_Winv), RHS)
    # Solve Sylvester Equation M̃H + H(W⁻¹HᵀC̃HW⁻¹) = C̃HW⁻¹
    # H .= sylvester(M_tilde, Winv_HᵀC̃H_Winv, RHS)
    # solve by eigendecomposition
    # λ_L = view(VC.storage_nn, (5d+1):(6d))
    # λ_R = view(VC.storage_nn, (6d+1):(6d+r))
    # _syev!('V', 'L', M_tilde, λ_L, VC.lworkspace, info)
    # Q_L = M_tilde
    # _syev!('V', 'L', Winv_HtC̃H_Winv, λ_R, VC.rworkspace, info)
    # Q_R = Winv_HtC̃H_Winv
    # mul!(H_Winv, transpose(Q_L), RHS)
    # mul!(RHS, H_Winv, Q_R)
    # for j in 1:r, i in 1:d
    #     H[i, j] = RHS[i, j] / (λ_L[i] + λ_R[j])
    # end
    # mul!(H_Winv, Q_L, H)
    # mul!(H, H_Winv, transpose(Q_R))
    # Sylvester end -----------------------------
    # Finally update F in-place, F = (√Ψ) H
    for j in axes(F, 2), i in axes(F, 1)
        F[i, j] = sqrtΨ[i] * H[i, j]
    end
    return nothing
end

function update_Σ!(VC::LowRankPlusDiagonal{T}, ::Val{:MM}) where {T}
    # F(n+1) is a function of F(n) and Ψ(n)
    # depends on Ψ(n), so F must be updated first to agree with the math
    update_F!(VC)
    # Ψ(n+1) is a function of Σ(n) which is not updated yet
    update_Ψ!(VC)
    # After updating F and Ψ, we materialize Σ(n+1) in place
    return materialize!(VC)
end

function rotate!(VC::LowRankPlusDiagonal{T}) where {T}
    F = VC.F
    d, r = size(F)
    storage_d = view(VC.storage_nn, 1:d, 1)
    if r > 1
        # Evaluate FᵀΨ⁻¹F
        Ψinv_F = view(VC.storage_nn, 1:d, 2:(r + 1))
        @inbounds for j in axes(F, 2), i in axes(F, 1)
            Ψinv_F[i, j] = F[i, j] / VC.Ψ[i]
        end
        Ft_Ψinv_F = view(VC.storage_nn, (d + 1):(d + r), 1:r)
        mul!(Ft_Ψinv_F, transpose(F), Ψinv_F) # Ψinv_F is free to be overwritten after this
        _syev!('V', 'L', Ft_Ψinv_F, storage_d, VC.rworkspace, VC.info)
        vecs = Ft_Ψinv_F
        reverse!(vecs; dims=2)
        mul!(F, copyto!(Ψinv_F, F), vecs)
    end
    return nothing
end

function finalize!(VC::LowRankPlusDiagonal{T}) where {T}
    return rotate!(VC)
end