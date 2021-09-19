module MultiResponseVarianceComponentModels

using LinearAlgebra, Manopt, Manifolds
import LinearAlgebra: BlasReal, copytri!
export fit!,
    kron_axpy!, 
    kron_reduction!, 
    loglikelihood!,
    MultiResponseVarianceComponentModel,
    update_res!,
    update_Ω!

struct MultiResponseVarianceComponentModel{T <: BlasReal}
    # data
    Y             :: Matrix{T}
    X             :: Matrix{T}
    V             :: Vector{Matrix{T}}
    # parameters
    Β             :: Matrix{T} # \Beta
    Σ             :: Vector{Matrix{T}}
    Ω             :: Matrix{T} # covariance Ω = Σ[1] ⊗ V[1] + ... + Σ[m] ⊗ V[m]
    Σ_rank        :: Vector{<:Integer}
    # working arrays
    R             :: Matrix{T} # residuals
    Ω⁻¹R          :: Matrix{T}
    xtx           :: Matrix{T} # Gram matrix X'X
    xty           :: Matrix{T} # X'Y
    storage_nd_1  :: Vector{T}
    storage_nd_2  :: Vector{T}
    storage_pd    :: Vector{T}
    storage_n_d   :: Matrix{T}
    storage_n_p   :: Matrix{T}
    storage_p_d   :: Matrix{T}
    storage_d_d_1 :: Matrix{T}
    storage_d_d_2 :: Matrix{T}
    storage_d_d_3 :: Matrix{T}
    storage_d_d_4 :: Matrix{T}
    storage_d_d_5 :: Matrix{T}
    storage_d_d_6 :: Matrix{T}
    storage_d_d_7 :: Matrix{T}
    storage_p_p   :: Matrix{T}
    storage_nd_nd :: Matrix{T}
    storage_pd_pd :: Matrix{T}
end

# constructor
function MultiResponseVarianceComponentModel(
    Y :: AbstractMatrix{T},
    X :: Union{Nothing, Matrix{T}},
    V :: Vector{<:AbstractMatrix{T}};
    Σ_rank :: Vector{<:Integer} = repeat([size(Y, 2)], inner = length(V))
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
    Β             = Matrix{T}(undef, p, d)
    Σ             = [Matrix{T}(undef, d, d) for _ in 1:m]
    Ω             = Matrix{T}(undef, nd, nd)
    # working arrays
    R             = Matrix{T}(undef, n, d)
    Ω⁻¹R          = Matrix{T}(undef, n, d)
    xtx           = transpose(X) * X
    xty           = transpose(X) * Y
    storage_nd_1  = Vector{T}(undef, nd)
    storage_nd_2  = Vector{T}(undef, nd)
    storage_pd    = Vector{T}(undef, pd)
    storage_n_d   = Matrix{T}(undef, n, d)
    storage_n_p   = Matrix{T}(undef, n, p)
    storage_p_d   = Matrix{T}(undef, p, d)
    storage_d_d_1 = Matrix{T}(undef, d, d)
    storage_d_d_2 = Matrix{T}(undef, d, d)
    storage_d_d_3 = Matrix{T}(undef, d, d)
    storage_d_d_4 = Matrix{T}(undef, d, d)
    storage_d_d_5 = Matrix{T}(undef, d, d)
    storage_d_d_6 = Matrix{T}(undef, d, d)
    storage_d_d_7 = Matrix{T}(undef, d, d)
    storage_p_p   = Matrix{T}(undef, p, p)
    storage_nd_nd = Matrix{T}(undef, nd, nd)
    storage_pd_pd = Matrix{T}(undef, pd, pd)
    MultiResponseVarianceComponentModel{T}(
        Y, Xmat, V,
        Β, Σ, Ω, Σ_rank,
        R, Ω⁻¹R, xtx, xty,
        storage_nd_1, storage_nd_2, storage_pd,
        storage_n_d, storage_n_p, storage_p_d,
        storage_d_d_1, storage_d_d_2, storage_d_d_3, 
        storage_d_d_4, storage_d_d_5, storage_d_d_6, storage_d_d_7,
        storage_p_p, storage_nd_nd, storage_pd_pd)
end

# univariate response case
MultiResponseVarianceComponentModel(y::AbstractVector, X, V) = 
    MultiResponseVarianceComponentModel(reshape(y, length(y), 1), X, V)

# no X case
MultiResponseVarianceComponentModel(Y::AbstractMatrix, V::Vector{<:AbstractMatrix}) = 
    MultiResponseVarianceComponentModel(Y, nothing, V)

include("multivariate_calculus.jl")
include("fit.jl")
include("manopt.jl")

end
