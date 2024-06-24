"""
    kron_axpy!(A, X, Y)

Overwrite `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)`, but more memory efficient.
"""
@inline function kron_axpy!(
    A :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    Y :: AbstractVecOrMat{T}
    ) where T <: Real
    m, n = size(A, 1), size(A, 2)
    p, q = size(X, 1), size(X, 2)
    @assert size(Y, 1) == m * p
    @assert size(Y, 2) == n * q
    yidx = 0
    @inbounds for j in 1:n, l in 1:q, i in 1:m
        aij = A[i, j]
        for k in 1:p
            Y[yidx += 1] += aij * X[k, l]
        end
    end
    Y
end

"""
    kron_reduction!(A, B, C; sym = false)

Overwrite `C` with the derivative of `tr(A' (X ⊗ B))` wrt `X`.
`C[i, j] = dot(Aij, B)`, where `Aij` is the `(i , j)` block of `A`. `sym = true` 
indicates `A` and `B` are symmetric.
"""
@inline function kron_reduction!(
    A   :: AbstractMatrix{T}, 
    B   :: AbstractMatrix{T},
    C   :: AbstractMatrix{T},
    sym :: Bool = false
    ) where T <: Real
    m, n = size(B)
    p, q = size(C)
    @assert size(A, 1) == m * p && size(A, 2) == n * q
    fill!(C, 0)
    # loop over (i, j) blocks of A, each of size B
    # aidx = 0
    # @inbounds for j in 1:q, l in 1:n, i in 1:p, k in 1:m
    #     C[i, j] += A[aidx += 1] * B[k, l]
    # end
    @inbounds for j in 1:q
        cidx = ((j - 1) * n + 1):(j * n)
        for i in 1:p
            if i ≤ j || ~sym
                C[i, j] = 0
                ridx    = ((i - 1) * m + 1):(i * m)
                bidx    = 0
                for c in cidx, r in ridx
                    C[i, j] += A[r, c] * B[bidx += 1]
                end
            end
        end
    end
    sym && copytri!(C, 'U')
    C
end

function duplication(n)
    D = zeros(Int, abs2(n), ◺(n))
    vechidx = 1
    for j in 1:n
        for i in j:n
            D[(j - 1) * n + i, vechidx] = 1
            D[(i - 1) * n + j, vechidx] = 1
            vechidx += 1
        end
    end
    D
end

"""
    vech(A::AbstractVecOrMat)

Return lower triangular part of `A`.
"""
vech(A::AbstractVecOrMat) = [A[i, j] for i in 1:size(A, 1), j in 1:size(A, 2) if i ≥ j]

function commutation(m::Int, n::Int)
    mn = m * n 
    reshape(kron(vec(Matrix{Float64}(I, m, m)), Matrix{Float64}(I, n, n)), mn, mn)
end

commutation(m::Int) = commutation(m, m)

"""
    ◺(n::Int)

Triangular number `n * (n + 1) / 2`.
"""
@inline ◺(n::Int) = (n * (n + 1)) >> 1