struct Unstructured{T} <: VarCompStructure{T}
    # parameters
    L             :: Matrix{T}
    pardim        :: Int
    # working arrays
    data          :: Matrix{T}
    datarank      :: Int
    storage_nd_nd :: Matrix{T}
    # Constants
    V             :: Matrix{T}
    Vrank         :: Int
end

function Unstructured(Σ::Matrix{T}, V::Matrix{T}) where {T<:BlasReal}
    n = LinearAlgebra.checksquare(V)
    d = LinearAlgebra.checksquare(Σ)
    nd = n * d
    data = copy(Σ)
    L = similar(Σ)
    LAPACK.potrf!('L', copyto!(L, data))
    tril!(L)
    pardim = ◺(d)
    datarank = rank(data)
    storage_nd_nd = Matrix{T}(undef, nd, nd)
    # avoid evaluation of rank(V), only provide it if needed
    Vrank = n
    return Unstructured{T}(L, pardim, data, datarank, storage_nd_nd, V, Vrank)
end

function Unstructured(d::Int, V::Matrix{T}) where {T<:BlasReal}
    # initialize with identity, overwritten later anyway
    data = Matrix{T}(UniformScaling{T}(1), d, d)
    # use constructor already defined
    Unstructured(data, V)
end