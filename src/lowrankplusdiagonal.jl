struct LowRankPlusDiagonal{T} <: VarCompStructure{T}
    # parameters
    F             :: Matrix{T}
    Ψ             :: Vector{T}
    pardim        :: Int
    # working arrays
    Σ             :: Matrix{T}
    Σrank         :: Int
    storage_nd_nd :: Matrix{T}
    storage_dd_dd :: Matrix{T}
    storage_nn    :: Matrix{T}
    storage_dd_1  :: Matrix{T}
    storage_dd_2  :: Matrix{T}
    storage_dd_3  :: Matrix{T}
    # Constants
    V             :: Matrix{T}
    Vrank         :: Int
end

function LowRankPlusDiagonal(F::Matrix{T}, Ψ::Vector{T}, V::Matrix{T}) where {T}
    n = LinearAlgebra.checksquare(V)
    d, r = size(F)
    nd = n * d
    (length(Ψ) == d) || throw(DimensionMismatch())
    Σ = Matrix{T}(undef, d, d)
    BLAS.syrk!('L', 'N', one(T), F, zero(T), Σ)
    copytri!(Σ, 'L')
    @inbounds for i in 1:d
        Σ[i, i] += Ψ[i]
    end
    # dimension here is the number of elements in augmented matrix [F ⋮ Ψ]
    pardim = d * (r + 1)
    # Here, `Σrank` is not the rank of Σ = FFᵀ + Ψ, but the number of columns in F
    Σrank = r
    storage_nd_nd = Matrix{T}(undef, nd, nd)
    storage_dd_dd = Matrix{T}(undef, abs2(d), abs2(d))
    storage_nn = Matrix{T}(undef, n, n)
    storage_dd_1 = Matrix{T}(undef, d, d)
    storage_dd_2 = Matrix{T}(undef, d, d)
    storage_dd_3 = Matrix{T}(undef, d, d)
    # avoid evaluation of rank(V), only provide it if needed
    Vrank = n
    return LowRankPlusDiagonal{T}(F, Ψ, pardim, 
                                  Σ,
                                  Σrank,
                                  storage_nd_nd,
                                  storage_dd_dd, 
                                  storage_nn,
                                  storage_dd_1,
                                  storage_dd_2,
                                  storage_dd_3,
                                  V, 
                                  Vrank)
end

function LowRankPlusDiagonal(d::Int, r::Int, V::Matrix{T}) where {T}
    F = Matrix{T}(undef, d, r)
    fill!(F, zero(T))
    dg = Vector{T}(one(T), d)
    LowRankPlusDiagonal(F, dg, V)
end

function LowRankPlusDiagonal(Σ::Matrix{T}, r::Int, V::Matrix{T}) where {T}
    # Best rank-r approximation to Σ
    d = LinearAlgebra.checksquare(Σ)
    # by convention, `eigen` is normally smallest to largest
    # permute the eigenvalues and eigenvectors to largest to smallest
    Σ_eig = eigen(Σ; sortby = x -> -abs(x))
    F = Matrix{T}(undef, d, r)
    for j in 1:r
        sqrtλ = sqrt(Σ_eig.values[j])
        for i in 1:d
            F[i, j] = sqrtλ * Σ_eig.vectors[i, j]
        end
    end
    # Initialize dg with the residual, diag(Σ - FFᵀ)
    dg = diag(Σ - F * transpose(F))
    # ensure initial point is positive
    dg .+= T(1e-6)
    LowRankPlusDiagonal(F, dg, V)
end

function initialize!(VC::LowRankPlusDiagonal{T}) where {T}
    Σ = VC.Σ
    # Best rank-r approximation to Σ
    d = LinearAlgebra.checksquare(Σ)
    r = VC.Σrank
    # by convention, `eigen` is normally smallest to largest
    # permute the eigenvalues and eigenvectors to largest to smallest
    Σ_eig = eigen(Σ; sortby = x -> -abs(x))
    F = VC.F
    for j in 1:r
        sqrtλ = sqrt(Σ_eig.values[j])
        for i in 1:d
            F[i, j] = sqrtλ * Σ_eig.vectors[i, j]
        end
    end
    # Initialize dg with the residual, diag(Σ - FFᵀ)
    # ensure initial point is positive
    VC.Ψ .= diag(Σ - F * transpose(F)) .+ T(1e-8)
    mul!(Σ, F, transpose(F), one(T), zero(T))
    for j in axes(Σ, 2)
        Σ[j, j] += VC.Ψ[j]
    end
    return nothing
end

@inline function update_Ψ!(VC::LowRankPlusDiagonal{T}) where {T}
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    Ψ = VC.Ψ
    rankedness = size(VC, 1)
    @inbounds for i in eachindex(Ψ)
        Ψ[i] = sqrt(N[i, i] / M[i, i]) * Ψ[i]
        rankedness -= (Ψ[i] < 1e-8)
    end
    rankedness < VC.Σrank && throw("Ψ has fewer than $(VC.Σrank) elements greater than 1e-8, model may be extremely ill-conditioned!")
    return Ψ
end

function update_F!(VC::LowRankPlusDiagonal{T}) where {T}
    d, r = size(VC.F)
    F = VC.F
    Ψ = VC.Ψ
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    C = VC.storage_dd_3
    # assume current estimate of Σ⁽ⁿ⁾ is stored in `VC.Σ` 
    # Intermediate array NΣ, can be immediately re-used
    NΣ = view(VC.storage_nn, 1:d, 1:d)
    BLAS.symm!('L', 'L', one(T), N, VC.Σ, zero(T), NΣ)
    # update C = ΣNΣ
    BLAS.symm!('L', 'L', one(T), VC.Σ, NΣ, zero(T), C)
    # Intermediate variable H = √Ψ \ F 
    sqrtΨ = view(VC.storage_nn, 1:d, 1)
    for i in 1:d
        sqrtΨ[i] = sqrt(Ψ[i])
    end
    # H = (√Ψ) \ F
    H = view(VC.storage_nn, 1:d, 2:(r + 1))
    for j in axes(F, 2), i in axes(F, 1)
        H[i, j] = F[i, j] / sqrtΨ[i]
    end
    # Intermediate array W = HᵀH + Iᵣ = FᵀΨ⁻¹F + Iᵣ
    W = view(VC.storage_nn, (d + 1):(d + r), 2:(r + 1))
    BLAS.syrk!('L', 'T', one(T), H, zero(T), W)
    for i in 1:r
        W[i, i] += 1
    end
    copytri!(W, 'L')
    # Factorize W since we need W⁻¹
    _, info = LAPACK.potrf!('L', W)
    info > 0 && throw(PosDefException(info))
    # Intermediate array M̃ = √Ψ Mₙ √Ψ
    M_tilde = view(VC.storage_nn, (d + r + 1):(d + r + d), 1:d)
    for j in axes(C, 2), i in axes(C, 1)
        M_tilde[i, j] = M[i, j] * sqrtΨ[j] * sqrtΨ[i]
    end
    # Intermediate array C̃ = (√Ψ)⁻¹ C (√Ψ)⁻¹
    C_tilde = view(VC.storage_nn, (2d + r + 1):(2d + r + d), 1:d)
    for j in axes(C, 2), i in axes(C, 1)
        C_tilde[i, j] = C[i, j] / (sqrtΨ[j] * sqrtΨ[i])
    end
    H_Winv = view(VC.storage_nn, (3d + r + 1):(3d + r + d), 1:r)
    BLAS.trsm!('R', 'L', 'T', 'N', one(T), W, copyto!(H_Winv, H))
    BLAS.trsm!('R', 'L', 'N', 'N', one(T), W, H_Winv)
    # space allocated to W can now be re-used
    # RHS of update equation
    RHS = view(VC.storage_nn, (d + r + 1):(d + r + d), (d + 1):(d + r))
    BLAS.symm!('L', 'L', one(T), C_tilde, H_Winv, zero(T), RHS)
    # Right-hand coefficient
    # Recycle space allocated to W
    Winv_HᵀC̃H_Winv = W
    mul!(Winv_HᵀC̃H_Winv, transpose(H_Winv), RHS)
    # Solve Sylvester Equation M̃ H + H (W⁻¹HᵀC̃HW⁻¹) = C̃HW⁻¹
    # H .= sylvester(M_tilde, Winv_HᵀC̃H_Winv, RHS)
    # solve by eigendecomposition
    λ_L, Q_L = LAPACK.syev!('V', 'L', M_tilde)
    λ_R, Q_R = LAPACK.syev!('V', 'L', Winv_HᵀC̃H_Winv)
    mul!(H_Winv, transpose(Q_L), RHS)
    mul!(RHS, H_Winv, Q_R)
    for j in 1:r, i in 1:d
        H[i, j] = RHS[i, j] / (λ_L[i] + λ_R[j])
    end
    mul!(H_Winv, Q_L, H)
    mul!(H, H_Winv, transpose(Q_R))
    # Finally update F in-place, F = √Ψ H
    for j in axes(F, 2), i in axes(F, 1)
        F[i, j] = sqrtΨ[i] * H[i, j]
    end
    return nothing
end

function update_Σ!(VC::LowRankPlusDiagonal{T}, ::Val{:MM}) where {T}
    # F(n+1) is a function of F(n) and Ψ(n)
    # depends on Ψ(n), so must be updated first to agree with the math
    update_F!(VC)
    update_Ψ!(VC)
    BLAS.syrk!('L', 'N', one(T), VC.F, zero(T), VC.Σ)
    @inbounds for i in axes(VC.Σ, 2)
        VC.Σ[i, i] += VC.Ψ[i]
    end
    return copytri!(VC.Σ, 'L')
end