"""
    vech!(vechA::AbstractVector{T}, A::AbstractMatrix{T})

Return the lower triangular part of `A`, updating `vechA` in-place.
"""

function vech!(vechA::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    n, m = size(A)
    (n != m) && throw(DimensionMismatch("A must be square.\n"))
    tn = ◺(n)
    (length(vechA) != tn) && throw(DimensionMismatch("vech(A) must be length $tn.\n"))
    idx::Int = 1
    @inbounds for j in 1:n, i in j:n
        vechA[idx] = A[i, j]
        idx += 1
    end
    return vechA
end

"""
    vech(A::AbstractMatrix{T})

Return the lower triangular part of `A` as a Vector{T}.
"""
function vech(A::AbstractMatrix{T}) where {T}
    n, m = size(A)
    tn = ◺(n)
    vechA = Vector{T}(undef, tn)
    return vech!(vechA, A)
end

"""
    unvech!(A::AbstractMatrix{T}, vechA::AbstractVector{T})

Fill the lower triangular part of `A` with `vechA`, unexported
"""
@inline function _unvech!(A::AbstractMatrix{T}, vechA::AbstractVector{T}) where {T}
    n = size(A, 1)
    idx::Int = 1
    @inbounds for j in 1:n, i in j:n
        A[i, j] = vechA[idx]
        idx += 1
    end
    return nothing
end


"""
   unvech!(A::AbstractMatrix{T}, vechA::AbstractVector{T})

Fill the lower triangular part of `A` with `vechA` in-place, returning `A`.
"""
function unvech!(A::AbstractMatrix{T}, vechA::AbstractVector{T}) where {T}
    n, m = size(A)
    (n != m) && throw(DimensionMismatch("A must be square.\n"))
    tn = ◺(n)
    (length(vechA) != tn) && throw(DimensionMismatch("vech(A) must be length $tn.\n"))
    _unvech!(A, vechA)
    return A
end

"""
    unvech(vechA::AbstractVector{T})

Return a square matrix with the lower triangular part filled from `vechA`.
"""
function unvech(vechA::AbstractVector{T}) where {T}
    tn = length(vechA)
    # Solve n(n+1)/2 = tn for n
    temp::T = (sqrt(8 * tn + 1) - 1) / 2
    # is the root an integer? if so, we can proceed
    if temp == trunc(temp)
        n = trunc(Int, temp)
        A = zeros(T, n, n)
        _unvech!(A, vechA)
    else
        throw(DimensionMismatch("vech(A) has an incompatible number of elements.\n"))
    end
    return A
end

# Some convenience functions for filling arrays with zeros or ones
@inline function zero!(x::AbstractArray{T}) where {T}
    fill!(x, zero(T))
end

@inline function zero!(x::Ref{T}) where {T}
    x[] = zero(T)
end

@inline function ones!(x::AbstractArray{T}) where {T}
    fill!(x, one(T))
end

@inline function ones!(x::Ref{T}) where {T}
    x[] = one(T)
end

"""
    blockcopytri!(S::AbstractMatrix{T}, m::Int, n::Int, uplo::AbstractChar)

Given a symmetric block matrix `S`, where S = U ⊗ V with U m by m and V n by n, 
copy the upper or lower triangular part of S to the other triangle, 
depending on `uplo`. This is useful for large block symmetric matrices, 
and is around twice as fast as `copytri!` on the full matrix. 
"""
# Useful for large block symmetric matrices, around twice as fast
@inline function blockcopytri!(
    S::AbstractMatrix{T}, 
    m::Int, 
    n::Int, 
    uplo::AbstractChar
    ) where {T}
    mn = checksquare(S)
    (mn != m * n) && throw(DimensionMismatch())
    @inbounds if uplo == 'U'
        for j in 1:m, i in 1:j
            rowidx = ((i - 1) * n + 1):(i*n)
            colidx = ((j - 1) * n + 1):(j*n)
            Sij = view(S, rowidx, colidx)
            if i == j
                copytri!(Sij, 'U')
            else
                Sji = view(S, colidx, rowidx)
                transpose!(Sji, Sij)
            end
        end
    elseif uplo == 'L'
        for j in 1:m, i in j:m
            rowidx = ((i - 1) * n + 1):(i*n)
            colidx = ((j - 1) * n + 1):(j*n)
            Sij = view(S, rowidx, colidx)
            if i == j
                copytri!(Sij, 'L')
            else
                Sji = view(S, colidx, rowidx)
                transpose!(Sji, Sij)
            end
        end
    else
        throw(ArgumentError(lazy"uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
    end
    return S
end

"""

"""
# syev! with all workspace defined
for (syev, elty) in ((:dsyev_, :Float64), (:ssyev_, :Float32))
    @eval begin
        function _syev_lwork!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
            Base.require_one_based_indexing(A)
            jobz in ('N', 'V') || throw(ArgumentError("jobz must be 'N' or 'V', got '$jobz'"))
            uplo in ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L', got '$uplo'"))
            n    = LinearAlgebra.checksquare(A)
            W    = Vector{$elty}(undef, n)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            ccall((@blasfunc($syev), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                  jobz, uplo, n, A, max(1, stride(A, 2)), W, work, BlasInt(-1), info, 1, 1)
            info[] < 0 && throw(ArgumentError("invalid argument #$(-info[]) to LAPACK call"))
            BlasInt(work[1])  # LAPACK writes optimal lwork into work[1] as a float
        end

        function _syev!(jobz::AbstractChar, 
                        uplo::AbstractChar,
                        A::AbstractMatrix{$elty},
                        W::AbstractVector{$elty},
                        work::AbstractVector{$elty}, 
                        info::Ref{BlasInt}
                        )
            Base.require_one_based_indexing(A, W, work)
            jobz in ('N', 'V') || throw(ArgumentError("jobz must be 'N' or 'V', got '$jobz'"))
            uplo in ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L', got '$uplo'"))
            stride(A,    1) == 1 || throw(ArgumentError("A must have stride 1 in the first dimension"))
            stride(W,    1) == 1 || throw(ArgumentError("W must have stride 1 in the first dimension"))
            stride(work, 1) == 1 || throw(ArgumentError("work must have stride 1 in the first dimension"))
            n    = LinearAlgebra.checksquare(A)
            # info = Ref{BlasInt}()
            ccall((@blasfunc($syev), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                  jobz, uplo, n, A, max(1, stride(A, 2)), W, work, BlasInt(length(work)), info, 1, 1)
            info[] < 0 && throw(ArgumentError("invalid argument #$(-info[]) to LAPACK call"))
            info[] > 0 && throw(ErrorException("LAPACK $($syev) failed to converge: info = $(info[])"))
            jobz == 'V' ? (W, A) : W
        end

        function _syevr_lwork!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
            Base.require_one_based_indexing(A)
            jobz ∈ ('N', 'V') || throw(ArgumentError("jobz must be 'N' or 'V', got '$jobz'"))
            uplo ∈ ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L', got '$uplo'"))
            n      = checksquare(A)
            W      = Vector{$elty}(undef, n)
            Z      = Matrix{$elty}(undef, jobz == 'V' ? n : 0, n)
            isuppz = Vector{BlasInt}(undef, 2n)
            work   = Vector{$elty}(undef, 1)
            iwork  = Vector{BlasInt}(undef, 1)
            info   = Ref{BlasInt}()
            m      = Ref{BlasInt}()
            ccall((@blasfunc($syevr), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                   Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Clong, Clong, Clong),
                  jobz, 'A', uplo, n,
                  A, max(1, stride(A, 2)), $elty(0), $elty(0),
                  BlasInt(0), BlasInt(0), $elty(-1), m,
                  W, Z, max(1, n), isuppz,
                  work, BlasInt(-1), iwork, BlasInt(-1),
                  info, 1, 1, 1)
            info[] < 0 && throw(ArgumentError("invalid argument #$(-info[]) to LAPACK call"))
            BlasInt(work[1]), BlasInt(iwork[1])
        end

        function _syevr!(jobz::AbstractChar, uplo::AbstractChar,
                         A::AbstractMatrix{$elty},
                         W::AbstractVector{$elty},
                         Z::AbstractMatrix{$elty},
                         isuppz::AbstractVector{BlasInt},
                         work::AbstractVector{$elty},
                         iwork::AbstractVector{BlasInt})
            Base.require_one_based_indexing(A, W, Z, isuppz, work, iwork)
            jobz ∈ ('N', 'V') || throw(ArgumentError("jobz must be 'N' or 'V', got '$jobz'"))
            uplo ∈ ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L', got '$uplo'"))
            n    = checksquare(A)
            m    = Ref{BlasInt}()
            info = Ref{BlasInt}()
            ccall((@blasfunc($syevr), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                   Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Clong, Clong, Clong),
                  jobz, 'A', uplo, n,
                  A, max(1, stride(A, 2)), $elty(0), $elty(0),
                  BlasInt(0), BlasInt(0), $elty(-1), m,
                  W, Z, max(1, n), isuppz,
                  work, BlasInt(length(work)), iwork, BlasInt(length(iwork)),
                  info, 1, 1, 1)
            info[] < 0 && throw(ArgumentError("invalid argument #$(-info[]) to LAPACK call"))
            info[] > 0 && throw(ErrorException("LAPACK $($syevr) failed to converge: info = $(info[])"))
            jobz == 'V' ? (W, Z) : W
        end
    end
end

