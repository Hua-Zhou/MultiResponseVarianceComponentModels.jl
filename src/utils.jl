function vech!(vechA::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    n, m = size(A)
    (n != m) && throw(DimensionMismatch("A must be square.\n"))
    tn = ◺(n)
    (length(vechA) != tn) && throw(DimensionMismatch("vech(A) must be length $tn.\n"))
    idx::Int = 1
    @inbounds for j in axes(A, 2), i in j:n
        vechA[idx] = A[i, j]
        idx += 1
    end
    return vechA
end

"""
    vech(A::AbstractVecOrMat)

Return lower triangular part of `A`.
"""
function vech(A::AbstractMatrix{T}) where {T}
    n, m = size(A)
    tn = ◺(n)
    vechA = Vector{T}(undef, tn)
    return vech!(vechA, A)
end

@inline function _unvech!(A::AbstractMatrix{T}, vechA::AbstractVector{T}) where {T}
    n = size(A, 1)
    idx::Int = 1
    @inbounds for j in axes(A, 2), i in j:n
        A[i, j] = vechA[idx]
        idx += 1
    end
    return nothing
end

function unvech!(A::AbstractMatrix{T}, vechA::AbstractVector{T}) where {T}
    n, m = size(A)
    (n != m) && throw(DimensionMismatch("A must be square.\n"))
    tn = ◺(n)
    (length(vechA) != tn) && throw(DimensionMismatch("vech(A) must be length $tn.\n"))
    _unvech!(A, vechA)
    return A
end

function unvech(vechA::AbstractVector{T}) where {T}
    tn = length(vechA)
    temp::T = (sqrt(8 * tn + 1) - 1) / 2
    if temp == trunc(temp)
        n = trunc(Int, temp)
        A = zeros(T, n, n)
        _unvech!(A, vechA)
    else
        throw(DimensionMismatch("vech(A) has an incompatible number of elements.\n"))
    end
    return A
end

@inline function zero!(x::AbstractArray{T}) where {T}
    fill!(x, zero(T))
end

@inline function zero!(x::Ref{T}) where {T}
    x[] = zero(T)
end

@inline function one!(x::AbstractArray{T}) where {T}
    fill!(x, one(T))
end

@inline function one!(x::Ref{T}) where {T}
    x[] = one(T)
end

# Useful for large block symmetric matrices, around twice as fast
@inline function blockcopytri!(
    X::AbstractMatrix{T}, 
    m::Int, 
    n::Int, 
    uplo::AbstractChar
    ) where {T}
    mn = LinearAlgebra.checksquare(X)
    (mn != m * n) && throw(DimensionMismatch())
    @inbounds if uplo == 'U'
        for j in 1:m, i in 1:j
            rowidx = ((i - 1) * n + 1):(i*n)
            colidx = ((j - 1) * n + 1):(j*n)
            Xij = view(X, rowidx, colidx)
            if i == j
                copytri!(Xij, 'U')
            else
                Xji = view(X, colidx, rowidx)
                transpose!(Xji, Xij)
            end
        end
    elseif uplo == 'L'
        for j in 1:m, i in j:m
            rowidx = ((i - 1) * n + 1):(i*n)
            colidx = ((j - 1) * n + 1):(j*n)
            Xij = view(X, rowidx, colidx)
            if i == j
                copytri!(Xij, 'L')
            else
                Xji = view(X, colidx, rowidx)
                transpose!(Xji, Xij)
            end
        end
    else
        throw(ArgumentError(lazy"uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
    end
    return X
end