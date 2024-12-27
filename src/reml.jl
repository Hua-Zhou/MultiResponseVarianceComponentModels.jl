function project_null(
    Y :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    V :: Vector{<:AbstractMatrix{T}}
    ) where {T <: Real}
    n, p, m = size(X, 1), size(X, 2), length(V)
    # basis of N(Xᵗ)
    Xᵗ = Matrix{T}(undef, p, n)
    transpose!(Xᵗ, X)
    A = nullspace(Xᵗ)
    s = size(A, 2) 
    Ỹ = transpose(A) * Y
    Ṽ = Vector{Matrix{T}}(undef, m)
    storage = zeros(n, s)
    for i in 1:m
        mul!(storage, V[i], A)
        Ṽ[i] = BLAS.gemm('T', 'N', A, storage)
    end
    Ỹ, Ṽ, A
end