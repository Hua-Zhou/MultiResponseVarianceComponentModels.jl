struct Unstructured{T} <: VarCompStructure{T}
    # parameters
    L             :: Matrix{T}
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

function Unstructured(Σ::Matrix{T}, V::Matrix{T}) where {T}
    n = LinearAlgebra.checksquare(V)
    d = LinearAlgebra.checksquare(Σ)
    nd = n * d
    Σ = copy(Σ)
    L = similar(Σ)
    LAPACK.potrf!('L', copyto!(L, Σ))
    tril!(L)
    pardim = ◺(d)
    Σrank = rank(Σ)
    storage_nd_nd = Matrix{T}(undef, nd, nd)
    storage_dd_dd = Matrix{T}(undef, abs2(d), abs2(d))
    storage_nn = Matrix{T}(undef, n, n)
    storage_dd_1 = Matrix{T}(undef, d, d)
    storage_dd_2 = similar(storage_dd_1)
    storage_dd_3 = similar(storage_dd_1)
    # avoid evaluation of rank(V), only provide it if needed
    Vrank = n
    return Unstructured{T}(L, pardim, 
                           Σ, Σrank,
                           storage_nd_nd, 
                           storage_dd_dd,
                           storage_nn,
                           storage_dd_1, 
                           storage_dd_2, 
                           storage_dd_3, 
                           V, 
                           Vrank)
end

function Unstructured(d::Int, V::Matrix{T}) where {T}
    # initialize with identity, overwritten later anyway
    Σ = Matrix{T}(UniformScaling{T}(1), d, d)
    # use constructor already defined
    Unstructured(Σ, V)
end

function initialize!(VC::Unstructured{T}) where {T}
    LAPACK.potrf!('L', copyto!(VC.L, VC.Σ))
    tril!(VC.L)
    return nothing
end

function update_Σ!(VC::Unstructured{T}, ::Val{:MM}) where {T}
    d = size(VC, 1)
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    C = VC.storage_dd_3
    # Assume M = tr[Ω⁻¹ ⊗ V] already stored
    # Lower Cholesky factor Lₘ, M = Lₘ Lₘᵀ
    Lₘ, info = LAPACK.potrf!('L', M)
    info > 0 && throw(PosDefException(info))
    # Assume N = ̃R̃ᵀVR̃ where R̃ = vec⁻¹(Ω⁻¹ vec(R)) stored
    # C = LᵀΣNΣL
    NΣ = view(VC.storage_nn, 1:d, 1:d)
    # assume current estimate of Σ⁽ⁿ⁾ is stored in `VC.Σ` 
    # Intermediate array NΣ, can be immediately re-used
    BLAS.symm!('L', 'L', one(T), N, VC.Σ, zero(T), NΣ)
    # update ΣNΣ
    BLAS.symm!('L', 'L', one(T), VC.Σ, NΣ, zero(T), C)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lₘ, C)
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), Lₘ, C)
    # Need symmetric square root of C, C small so syevr! should be fine
    # eigenvalue upper bound by Gershgorin Circle Theorem
    radii = view(VC.storage_nn, 1:d)
    for j in axes(C, 2), i in axes(C, 1)
        if i != j 
            radii[j] += abs(C[i, j])
        end
    end
    # LAPACK.syevr! is allocating, which is annoying
    vals, vecs = LAPACK.syevr!('V', 'A', 'L', C, zero(T), maximum(radii), 1, d, T(1e-8))
    for j in eachindex(vals)
        vj = view(vecs, :, j)
        if vals[j] > 0
            quadroot = sqrt(sqrt(vals[j]))
            vj .*= quadroot
        else
            fill!(vj, zero(T))
        end
    end
    BLAS.syrk!('L', 'N', one(T), vecs, zero(T), VC.Σ)
    copytri!(VC.Σ, 'L')
    # Right multiply by L⁻¹
    BLAS.trsm!('R', 'L', 'N', 'N', one(T), Lₘ, VC.Σ)
    # Left multiply by L⁻ᵀ
    BLAS.trsm!('L', 'L', 'T', 'N', one(T), Lₘ, VC.Σ)
    # Solve for parameters L
    LAPACK.potrf!('L', copyto!(VC.L, VC.Σ))
    tril!(VC.L)
    return VC.Σ
end

function update_Σ!(VC::Unstructured{T}, ::Val{:EM}) where {T}
    d = size(VC, 1)
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    C = VC.storage_dd_3
    C .= (N .- M) ./ VC.Σrank
    mul!(M, C, VC.Σ)
    for j in 1:d
        M[j, j] += 1
    end
    mul!(VC.Σ, copyto!(N, VC.Σ), M)
    # Project to 𝕊
    transpose!(C, VC.Σ)
    VC.Σ .= (VC.Σ .+ C) ./ 2
    # Solve for parameters L
    LAPACK.potrf!('L', copyto!(VC.L, VC.Σ))
    tril!(VC.L)
    return VC.Σ
end