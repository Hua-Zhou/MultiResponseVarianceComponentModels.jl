"""
    kron_axpy!(A, X, Y)

Overwrite `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)` but
more memory efficient.
"""
function kron_axpy!(
    A :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    Y :: AbstractVecOrMat{T}
    ) where T <: Real
    m, n = size(A)
    p, q = size(X)
    @assert size(Y, 1) == m * p
    @assert size(Y, 2) == n * q
    @inbounds for j in 1:n
        coffset = (j - 1) * q
        for i in 1:m
            a = A[i, j]
            roffset = (i - 1) * p            
            for l in 1:q
                r = roffset + 1
                c = coffset + l
                for k in 1:p                
                    Y[r, c] += a * X[k, l]
                    r += 1
                end
            end
        end
    end
    Y
end

"""
    kron_reduction!(A, B, C, sym = true)

Overwrites `C` with the derivative of `tr(A' (X ⊗ B))` wrt `X`.
`C[i, j] = dot(Aij, B)`, where `Aij` is the `(i , j)` block of `A`. `sym=true` 
indicates `A` and `B` are symmetric.
"""
function kron_reduction!(
    A   :: AbstractMatrix{T}, 
    B   :: AbstractMatrix{T},
    C   :: AbstractMatrix{T},
    sym :: Bool = false
    ) where T <: Real
    # retrieve matrix sizes
    m, n = size(B)
    p, q = size(C)
    @assert size(A, 1) == m * p && size(A, 2) == n * q
    # loop over (i, j) blocks of A, each of size B
    for j in 1:q
        cidx = ((j - 1) * n + 1):(j * n)
        for i in 1:p
            if i ≥ j || ~sym
                ridx    = ((i - 1) * m + 1):(i * m)            
                Aij     = view(A, ridx, cidx)
                C[i, j] = dot(B, Aij)
            end
        end
    end
    sym && copytri!(C, 'L')
    C
end
