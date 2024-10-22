struct LowRankPlusDiagonal{T} <: VarCompStructure{T}
    # parameters
    F             :: Matrix{T}
    Ψ             :: Vector{T}
    pardim        :: Int
    # working arrays
    Σ             :: Matrix{T}
    Σrank         :: Int
    storage_nd_nd :: Matrix{T}
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
    dg = Vector{T}(undef, d)
    for i in 1:d
        dg[i] = Σ[i, i]
    end
    dg .-= sum(abs2, F; dims = 2)
    # ensure initial point is positive
    dg .+= T(1e-4)
    LowRankPlusDiagonal(F, dg, V)
end

function update_Ψ!(VC::LowRankPlusDiagonal{T}) where {T}
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    Ψ = VC.Ψ
    for i in eachindex(Ψ)
        Ψ[i] = sqrt(N[i, i] / M[i, i]) * Ψ[i]
    end
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
    # update C = ΣₙNₙΣₙ
    BLAS.symm!('L', 'L', one(T), VC.Σ, NΣ, zero(T), C)
    # Intermediate variable H = √Ψ \ F 
    sqrtΨ = view(VC.storage_nn, 1:d, 1)
    for i in 1:d
        sqrtΨ[i] = sqrt(Ψ[i])
    end
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
    _, info = LAPACK.potrf!('L', W)
    info > 0 && throw(PosDefException(info))
    # Intermediate array M_tilde = √Ψ Mₙ √Ψ
    M_tilde = view(VC.storage_nn, (d + r + 1):(2d + r), 1:d)
    for j in axes(C, 2), i in axes(C, 1)
        M_tilde[i, j] = M[i, j] * sqrtΨ[j] * sqrtΨ[i]
    end
    # Intermediate array (√Ψ \ C) / √Ψ
    C_tilde = view(VC.storage_nn, (d + r + 1):(2d + r), (d + 1):(2d))
    for j in axes(C, 2), i in axes(C, 1)
        C_tilde[i, j] = C[i, j] / (sqrtΨ[j] * sqrtΨ[i])
    end
    H_Winv = view(VC.storage_nn, (d + r + 1):(2d + r), (2d + 1):(2d + r))
    BLAS.trsm!('R', 'L', 'T', 'N', one(T), W, copyto!(H_Winv, H))
    BLAS.trsm!('R', 'L', 'N', 'N', one(T), W, H_Winv)
    # space allocated to W can now be re-used
    # RHS of update equation
    RHS = view(VC.storage_nn, (d + r + 1):(2d + r), (2d + r + 1):(2d + 2r))
    BLAS.symm!('L', 'L', one(T), C_tilde, H_Winv, zero(T), RHS)
    # Right-side coefficient
    mul!(W, transpose(H_Winv), RHS)
    # Solve Sylvester Equation
    # TODO: allocation-free
    H .= sylvester(M_tilde, W, RHS)
    for j in axes(F, 2), i in axes(F, 1)
        F[i, j] = H[i, j] * sqrtΨ[i]
    end
    return nothing
end

function update_Σ!(VC::LowRankPlusDiagonal{T}) where {T}
    update_F!(VC)
    update_Ψ!(VC)
    fill!(VC.Σ, zero(T))
    for i in axes(VC.Σ, 1)
        VC.Σ[i, i] = VC.Ψ[i]
    end
    BLAS.syrk!('L', 'N', one(T), VC.F, one(T), VC.Σ)
    copytri!(VC.Σ, 'L')
end