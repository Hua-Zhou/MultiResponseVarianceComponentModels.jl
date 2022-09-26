module MultiResponseVarianceComponentModels

using IterativeSolvers, LinearAlgebra, Manopt, Manifolds, Distributions
import LinearAlgebra: BlasReal, copytri!
export fit!,
    kron_axpy!, 
    kron_reduction!, 
    loglikelihood!,
    MultiResponseVarianceComponentModel,
    update_res!,
    update_Ω!,
    fisher_Σ!,
    lrt,
    h2,
    rg

struct MultiResponseVarianceComponentModel{T <: BlasReal}
    # data
    Y                :: Matrix{T}
    X                :: Matrix{T}
    V                :: Vector{Matrix{T}}
    # parameters
    B                :: Matrix{T}
    Σ                :: Vector{Matrix{T}}
    Ω                :: Matrix{T} # covariance Ω = Σ[1] ⊗ V[1] + ... + Σ[m] ⊗ V[m]
    Σ_rank           :: Vector{Int}
    # working arrays
    V_rank           :: Vector{Int}
    R                :: Matrix{T} # residuals
    Ω⁻¹R             :: Matrix{T}
    xtx              :: Matrix{T} # Gram matrix X'X
    xty              :: Matrix{T} # X'Y
    storage_nd_1     :: Vector{T}
    storage_nd_2     :: Vector{T}
    storage_pd       :: Vector{T}
    storage_n_d      :: Matrix{T}
    storage_n_p      :: Matrix{T}
    storage_p_d      :: Matrix{T}
    storage_d_d_1    :: Matrix{T}
    storage_d_d_2    :: Matrix{T}
    storage_d_d_3    :: Matrix{T}
    storage_d_d_4    :: Matrix{T}
    storage_d_d_5    :: Matrix{T}
    storage_d_d_6    :: Matrix{T}
    storage_d_d_7    :: Matrix{T}
    storage_p_p      :: Matrix{T}
    storage_nd_nd    :: Matrix{T}
    storage_pd_pd    :: Matrix{T}
    storage_nd_n_1   :: Matrix{T}
    storage_nd_n_2   :: Matrix{T}
    storage_nd_n_3   :: Matrix{T}
    Bcov             :: Matrix{T}
    Σcov             :: Matrix{T}
    logl             :: Vector{T}
end

# constructor
function MultiResponseVarianceComponentModel(
    Y :: AbstractMatrix{T},
    X :: Union{Nothing, Matrix{T}},
    V :: Vector{<:AbstractMatrix{T}};
    Σ_rank :: Vector{<:Integer} = fill(size(Y, 2), length(V))
    ) where T <: BlasReal
    # dimensions
    n, d, m = size(Y, 1), size(Y, 2), length(V)
    if X === nothing
        p = 0
        Xmat = Matrix{T}(undef, n, 0)
    else
        Xmat = X
        p = size(X, 2)
    end
    nd, pd = n * d, p * d
    # parameters
    B                = Matrix{T}(undef, p, d)
    Σ                = [Matrix{T}(undef, d, d) for _ in 1:m]
    Ω                = Matrix{T}(undef, nd, nd)
    V_rank           = [rank(V[k]) for k in 1:m]
    # working arrays
    R                = Matrix{T}(undef, n, d)
    Ω⁻¹R             = Matrix{T}(undef, n, d)
    xtx              = transpose(Xmat) * Xmat
    xty              = transpose(Xmat) * Y
    storage_nd_1     = Vector{T}(undef, nd)
    storage_nd_2     = Vector{T}(undef, nd)
    storage_pd       = Vector{T}(undef, pd)
    storage_n_d      = Matrix{T}(undef, n, d)
    storage_n_p      = Matrix{T}(undef, n, p)
    storage_p_d      = Matrix{T}(undef, p, d)
    storage_d_d_1    = Matrix{T}(undef, d, d)
    storage_d_d_2    = Matrix{T}(undef, d, d)
    storage_d_d_3    = Matrix{T}(undef, d, d)
    storage_d_d_4    = Matrix{T}(undef, d, d)
    storage_d_d_5    = Matrix{T}(undef, d, d)
    storage_d_d_6    = Matrix{T}(undef, d, d)
    storage_d_d_7    = Matrix{T}(undef, d, d)
    storage_p_p      = Matrix{T}(undef, p, p)
    storage_nd_nd    = Matrix{T}(undef, nd, nd)
    storage_pd_pd    = Matrix{T}(undef, pd, pd)
    storage_nd_n_1   = Matrix{T}(undef, nd, n)
    storage_nd_n_2   = Matrix{T}(undef, nd, n)
    storage_nd_n_3   = Matrix{T}(undef, nd, n)
    Bcov             = Matrix{T}(undef, pd, pd)
    Σcov             = Matrix{T}(undef, m * (binomial(d, 2) + d), m * (binomial(d, 2) + d))
    logl             = zeros(T, 1)
    MultiResponseVarianceComponentModel{T}(
        Y, Xmat, V,
        B, Σ, Ω, Σ_rank, V_rank,
        R, Ω⁻¹R, xtx, xty,
        storage_nd_1, storage_nd_2, storage_pd,
        storage_n_d, storage_n_p, storage_p_d,
        storage_d_d_1, storage_d_d_2, storage_d_d_3, 
        storage_d_d_4, storage_d_d_5, storage_d_d_6, storage_d_d_7,
        storage_p_p, storage_nd_nd, storage_pd_pd,
        storage_nd_n_1, storage_nd_n_2, storage_nd_n_3,
        Bcov, Σcov, logl)
end

MultiResponseVarianceComponentModel(Y::AbstractMatrix, x::AbstractVector, V::Vector{<:AbstractMatrix}) = 
    MultiResponseVarianceComponentModel(Y, reshape(x, length(x), 1), V)

MultiResponseVarianceComponentModel(y::AbstractVector, X::AbstractMatrix, V::Vector{<:AbstractMatrix}) = 
    MultiResponseVarianceComponentModel(reshape(y, length(y), 1), X, V)

MultiResponseVarianceComponentModel(y::AbstractVector, x::AbstractVector, V::Vector{<:AbstractMatrix}) = 
    MultiResponseVarianceComponentModel(reshape(y, length(y), 1), reshape(x, length(x), 1), V)

MultiResponseVarianceComponentModel(Y::AbstractMatrix, V::Vector{<:AbstractMatrix}) = 
    MultiResponseVarianceComponentModel(Y, nothing, V)

MultiResponseVarianceComponentModel(y::AbstractVector, V::Vector{<:AbstractMatrix}) = 
    MultiResponseVarianceComponentModel(reshape(y, length(y), 1), nothing, V)

MultiResponseVarianceComponentModel(Y, X, V::AbstractMatrix) =
    MultiResponseVarianceComponentModel(Y, X, [V])

MultiResponseVarianceComponentModel(Y, V::AbstractMatrix) =
    MultiResponseVarianceComponentModel(Y, [V])

function Base.show(io::IO, model::MultiResponseVarianceComponentModel)
    n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.V)
    if d == 1
        print(io, "A univariate response model (n = $n, p = $p, m = $m)")
    else
        print(io, "A multivariate response model (n = $n, d = $d, p = $p, m = $m)")
    end
end
    
include("multivariate_calculus.jl")
include("fit.jl")
include("manopt.jl")

end