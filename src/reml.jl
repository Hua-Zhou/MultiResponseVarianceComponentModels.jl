function project_null(
    Y :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    V :: Vector{<:AbstractMatrix{T}}
    ) where {T <: Real}
    n, p, m = size(X, 1), size(X, 2), length(V)
    if isempty(X)
        Y, V, Matrix{T}(I, n, n)
    else
        # basis of N(X')
        Xt = Matrix{T}(undef, size(X, 2), size(X, 1))
        transpose!(Xt, X)
        A = nullspace(Xt)
        s = size(A, 2) 
        Ỹ = A' * Y
        Ṽ = Vector{Matrix{T}}(undef, m)
        storage = zeros(n, s)
        for i in 1:m
            mul!(storage, V[i], A)
            Ṽ[i] = BLAS.gemm('T', 'N', A, storage)
        end 
        Ỹ, Ṽ, A
    end 
end