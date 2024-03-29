module MultiResponseVarianceComponentModels

using IterativeSolvers, LinearAlgebra, Manopt, Manifolds, Distributions
import LinearAlgebra: BlasReal, copytri!
export fit!,
    kron_axpy!, 
    kron_reduction!, 
    loglikelihood!,
    MultiResponseVarianceComponentModel,
    MRVCModel,
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
    Γ                :: Vector{Matrix{T}} # for manopt.jl
    Ψ                :: Matrix{T} # for manopt.jl
    Σ_rank           :: Vector{Int} # for manopt.jl
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
    storage_d_d_4    :: Matrix{T} # for manopt.jl
    storage_d_d_5    :: Matrix{T} # for manopt.jl
    storage_d_d_6    :: Matrix{T} # for manopt.jl
    storage_d_d_7    :: Matrix{T} # for manopt.jl
    storage_p_p      :: Matrix{T}
    storage_nd_nd    :: Matrix{T}
    storage_pd_pd    :: Matrix{T}
    storages_nd_nd   :: Vector{Matrix{T}} # for fisher_Σ!
    Bcov             :: Matrix{T} # for fisher_B!
    Σcov             :: Matrix{T} # for fisher_Σ!
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
    Γ                = [Matrix{T}(undef, d, Σ_rank[k]) for k in 1:m]
    Ψ                = Matrix{T}(undef, d, m)
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
    storages_nd_nd   = [Matrix{T}(undef, nd, nd) for _ in 1:m]
    Bcov             = Matrix{T}(undef, pd, pd)
    Σcov             = Matrix{T}(undef, m * (binomial(d, 2) + d), m * (binomial(d, 2) + d))
    logl             = zeros(T, 1)
    MultiResponseVarianceComponentModel{T}(
        Y, Xmat, V,
        B, Σ, Ω, Γ, Ψ, Σ_rank,
        V_rank, R, Ω⁻¹R, xtx, xty,
        storage_nd_1, storage_nd_2, storage_pd,
        storage_n_d, storage_n_p, storage_p_d,
        storage_d_d_1, storage_d_d_2, storage_d_d_3, 
        storage_d_d_4, storage_d_d_5, storage_d_d_6, storage_d_d_7,
        storage_p_p, storage_nd_nd, storage_pd_pd,
        storages_nd_nd, Bcov, Σcov, logl)
end

const MRVCModel = MultiResponseVarianceComponentModel

MRVCModel(Y::AbstractMatrix, x::AbstractVector, V::Vector{<:AbstractMatrix}) = 
    MRVCModel(Y, reshape(x, length(x), 1), V)

MRVCModel(y::AbstractVector, X::AbstractMatrix, V::Vector{<:AbstractMatrix}) = 
    MRVCModel(reshape(y, length(y), 1), X, V)

MRVCModel(y::AbstractVector, x::AbstractVector, V::Vector{<:AbstractMatrix}) = 
    MRVCModel(reshape(y, length(y), 1), reshape(x, length(x), 1), V)

MRVCModel(Y::AbstractMatrix, V::Vector{<:AbstractMatrix}) = MRVCModel(Y, nothing, V)

MRVCModel(y::AbstractVector, V::Vector{<:AbstractMatrix}) = 
    MRVCModel(reshape(y, length(y), 1), nothing, V)

MRVCModel(Y, X, V::AbstractMatrix) = MRVCModel(Y, X, [V])

MRVCModel(Y, V::AbstractMatrix) = MRVCModel(Y, [V])

function Base.show(io::IO, model::MultiResponseVarianceComponentModel)
    n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.V)
    if d == 1
        printstyled(io, "A univariate response variance component model\n"; underline = true)
    elseif d == 2
        printstyled(io, "A bivariate response variance component model\n"; underline = true)
    else
        printstyled(io, "A multivariate response variance component model\n"; underline = true)
    end
    print(io, "   * number of responses: ")
    printstyled(io, "$d\n"; color = :yellow)
    print(io, "   * number of observations: ")
    printstyled(io, "$n\n"; color = :yellow)
    print(io, "   * number of fixed effects: ")
    printstyled(io, "$p\n"; color = :yellow)
    print(io, "   * number of variance components: ")
    printstyled(io, "$m"; color = :yellow)
end

include("multivariate_calculus.jl")
include("fit.jl")
include("manopt.jl")
include("parse.jl")

end