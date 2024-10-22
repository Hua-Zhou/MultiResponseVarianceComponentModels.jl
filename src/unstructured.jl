struct Unstructured{T} <: VarCompStructure{T}
    # parameters
    L             :: Matrix{T}
    pardim        :: Int
    # working arrays
    Σ             :: Matrix{T}
    Σrank         :: Int
    storage_nd_nd :: Matrix{T}
    storage_nn    :: Matrix{T}
    storage_dd_1  :: Matrix{T}
    storage_dd_2  :: Matrix{T}
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
    storage_dd_1 = Matrix{T}(undef, d, d)
    storage_dd_2 = Matrix{T}(undef, d, d)
    # avoid evaluation of rank(V), only provide it if needed
    Vrank = n
    return Unstructured{T}(L, pardim, Σ, Σrank, storage_nd_nd, storage_dd_1, storage_dd_2, V, Vrank)
end

function Unstructured(d::Int, V::Matrix{T}) where {T}
    # initialize with identity, overwritten later anyway
    Σ = Matrix{T}(UniformScaling{T}(1), d, d)
    # use constructor already defined
    Unstructured(Σ, V)
end