"""
    kron_axpy!(A, X, Y)

Overwrite `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)`, but more memory efficient.
"""
@inline function kron_axpy!(
    A :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    Y :: AbstractVecOrMat{T}
    ) where {T}
    m, n = size(A)
    p, q = size(X)
    mp, nq = size(Y)
    (mp == m * p && nq == n * q) || throw(DimensionMismatch())
    yidx = firstindex(Y)
    @inbounds for j in axes(A, 2), l in axes(X, 2), i in axes(A, 1)
        aij = A[i, j]
        if !iszero(aij)
            for k in axes(X, 1)
                Y[yidx] += aij * X[k, l]
                yidx += 1
            end
        else
            # if aᵢⱼ == 0, can skip that block of calculations
            yidx += p
        end
    end
    return Y
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
    ) where {T}
    m, n = size(B)
    p, q = size(C)
    @assert size(A, 1) == m * p && size(A, 2) == n * q
    fill!(C, zero(T))
    # loop over (i, j) blocks of A, each of size B
    # aidx = 0
    # @inbounds for j in 1:q, l in 1:n, i in 1:p, k in 1:m
    #     C[i, j] += A[aidx += 1] * B[k, l]
    # end
    # @inbounds for j in 1:q
    #     cidx = ((j - 1) * n + 1):(j * n)
    #     for i in 1:p
    #         if i ≤ j || ~sym
    #             C[i, j] = 0
    #             ridx    = ((i - 1) * m + 1):(i * m)
    #             bidx    = 0
    #             for c in cidx, r in ridx
    #                 C[i, j] += A[r, c] * B[bidx += 1]
    #             end
    #         end
    #     end
    # end
    # above code is less efficient for large matrices A
    # easier and simpler to just iterate over blocks
    for j in axes(C, 2), i in axes(C, 1)
        # fill in upper triangle of C
        rstartidx = (i - 1) * p + 1 
        rendidx = i * p
        cstartidx = (j - 1) * q + 1
        cendidx = j * q
        if i ≤ j || sym == false
            C[i, j] = dot(view(A, rstartidx:rendidx, cstartidx:cendidx), B)
        end
    end
    sym && copytri!(C, 'U')
    C
end

function duplication(n::Int)
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

function commutation(m::Int, n::Int)
    K = zeros(Int, m * n, m * n)
    colK = 1
    @inbounds for j in 1:n, i in 1:m
        rowK          = n * (i - 1) + j
        K[rowK, colK] = 1
        colK += 1
    end
    K
end

commutation(m::Int) = commutation(m, m)

"""
    ◺(n::Int)

Triangular number `n * (n + 1) / 2`.
"""
@inline ◺(n::Int) = (n * (n + 1)) >> 1