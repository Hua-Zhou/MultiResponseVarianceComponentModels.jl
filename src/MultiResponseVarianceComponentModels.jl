module MultiResponseVarianceComponentModels

using IterativeSolvers, LinearAlgebra, Manopt, Manifolds, Distributions
import LinearAlgebra: BlasReal, copytri!
export fit!,
    kron_axpy!, 
    kron_reduction!, 
    loglikelihood!,
    MRVCModel,
    MultiResponseVarianceComponentModel,
    update_res!,
    update_Ω!,
    fisher_Σ!,
    lrt,
    h2,
    rg,
    permute

struct MRVCModel{T <: BlasReal}
    # data
    Y                  :: Matrix{T}
    X                  :: Matrix{T}
    V                  :: Vector{Matrix{T}}
    # parameters
    B                  :: Matrix{T}
    Σ                  :: Vector{Matrix{T}}
    Ω                  :: Matrix{T} # covariance Ω = Σ[1] ⊗ V[1] + ... + Σ[m] ⊗ V[m]
    Γ                  :: Vector{Matrix{T}} # for manopt.jl
    Ψ                  :: Matrix{T}         # for manopt.jl
    Σ_rank             :: Vector{Int}       # for manopt.jl
    # working arrays
    V_rank             :: Vector{Int}
    R                  :: Matrix{T} # residuals
    Ω⁻¹R               :: Matrix{T}
    xtx                :: Matrix{T} # Gram matrix X'X
    xty                :: Matrix{T} # X'Y
    storage_nd_1       :: Vector{T}
    storage_nd_2       :: Vector{T}
    storage_pd         :: Vector{T}
    storage_n_d        :: Matrix{T}
    storage_n_p        :: Matrix{T}
    storage_p_d        :: Matrix{T}
    storage_d_d_1      :: Matrix{T}
    storage_d_d_2      :: Matrix{T}
    storage_d_d_3      :: Matrix{T}
    storage_d_d_4      :: Matrix{T} # for manopt.jl
    storage_d_d_5      :: Matrix{T} # for manopt.jl
    storage_d_d_6      :: Matrix{T} # for manopt.jl
    storage_d_d_7      :: Matrix{T} # for manopt.jl
    storage_p_p        :: Matrix{T}
    storage_nd_nd      :: Matrix{T}
    storage_pd_pd      :: Matrix{T}
    logl               :: Vector{T} # likelihood
    storages_nd_nd     :: Union{Nothing, Vector{Matrix{T}}} # for fisher_Σ!
    Bcov               :: Union{Nothing, Matrix{T}} # for fisher_B!
    Σcov               :: Union{Nothing, Matrix{T}} # for fisher_Σ!
    # permutation for missing response
    P                  :: Union{Nothing, Vector{T}}
    invP               :: Union{Nothing, Vector{T}}
    # working arrays for missing response
    storage_nd_miss    :: Union{Nothing, Vector{T}}
    storage_nd_nd_miss :: Union{Nothing, Matrix{T}}
    # original data for reml
    Y_reml             :: Union{Nothing, Matrix{T}}
    X_reml             :: Union{Nothing, Matrix{T}}
    V_reml             :: Union{Nothing, Vector{Matrix{T}}}
    # fixed effects parameters for reml
    B_reml             :: Union{Nothing, Matrix{T}}
    # working arrays for reml
    Ω_reml             :: Union{Nothing, Matrix{T}}
    R_reml             :: Union{Nothing, Matrix{T}}
    storage_nd_nd_reml :: Union{Nothing, Matrix{T}}
    storage_pd_pd_reml :: Union{Nothing, Matrix{T}}
    storage_n_p_reml   :: Union{Nothing, Matrix{T}}
    storage_nd_1_reml  :: Union{Nothing, Vector{T}}
    storage_nd_2_reml  :: Union{Nothing, Vector{T}}
    storage_n_d_reml   :: Union{Nothing, Matrix{T}}
    storage_p_d_reml   :: Union{Nothing, Matrix{T}}
    storage_pd_reml    :: Union{Nothing, Vector{T}}
    logl_reml          :: Union{Nothing, Vector{T}}
    Bcov_reml          :: Union{Nothing, Matrix{T}}
    se                 :: Bool
    reml               :: Bool
end

# constructor
function MRVCModel(
    Y :: AbstractMatrix{T},
    X :: Union{Nothing, Matrix{T}},
    V :: Vector{<:AbstractMatrix{T}};
    Σ_rank :: Vector{<:Integer} = fill(size(Y, 2), length(V)),
    se :: Bool = true,
    reml :: Bool = false
    ) where T <: BlasReal
    # dimensions
    if any(ismissing, Y)
        @assert reml == false
        @assert se == false
        P, invP            = permute(Y)
        storage_nd_miss    = Vector{T}(undef, nd)
        storage_nd_nd_miss = Matrix{T}(undef, nd, nd)
    else
        P = invP = storage_nd_miss = storage_nd_nd_miss = nothing
    end
    if X === nothing
        p = 0
        reml = false
        Xmat = Matrix{T}(undef, n, 0)
    else
        Xmat = X
        p = size(X, 2)
    end
    d, m = size(Y, 2), length(V)
    if reml
        Y_reml = deepcopy(Y)
        X_reml = deepcopy(Xmat)
        V_reml = deepcopy(V)
        n_reml, p_reml, p = size(Y_reml, 1), size(X_reml, 2), 0
        nd_reml = n_reml * d
        pd_reml = p_reml * d
        B_reml  = Matrix{T}(undef, p_reml, d)
        Ω_reml  = Matrix{T}(undef, nd_reml, nd_reml)
        R_reml  = Matrix{T}(undef, n_reml, d)
        storage_nd_nd_reml = Matrix{T}(undef, nd_reml, nd_reml)
        storage_pd_pd_reml = Matrix{T}(undef, pd_reml, pd_reml)
        storage_n_p_reml = Matrix{T}(undef, n_reml, p_reml)
        storage_nd_1_reml = Vector{T}(undef, nd_reml)
        storage_nd_2_reml = Vector{T}(undef, nd_reml)
        storage_n_d_reml = Matrix{T}(undef, n_reml, d)
        storage_p_d_reml = Matrix{T}(undef, p_reml, d)
        storage_pd_reml = Vector{T}(undef, pd_reml)
        logl_reml = zeros(T, 1)
        Y, V, _ = project_null(Y_reml, X_reml, V_reml)
        Xmat = Matrix{T}(undef, size(Y, 1), 0)
    else
        Y_reml = X_reml = V_reml = B_reml = R_reml = Ω_reml =
            storage_nd_nd_reml = storage_pd_pd_reml = 
            storage_n_p_reml = storage_nd_1_reml = 
            storage_nd_2_reml = storage_n_d_reml = 
            storage_p_d_reml = storage_pd_reml = 
            logl_reml = Bcov_reml = nothing
    end
    n = size(Y, 1)
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
    logl             = zeros(T, 1)
    if se
        storages_nd_nd = [Matrix{T}(undef, nd, nd) for _ in 1:m]
        Bcov           = Matrix{T}(undef, pd, pd)
        Σcov           = Matrix{T}(undef, m * (binomial(d, 2) + d), m * (binomial(d, 2) + d))
        if reml
            Bcov_reml  = Matrix{T}(undef, pd_reml, pd_reml)
        end
    else
        storages_nd_nd = Bcov = Σcov = Bcov_reml = nothing
    end
    MRVCModel{T}(
        Y, Xmat, V,
        B, Σ, Ω, Γ, Ψ, Σ_rank,
        V_rank, R, Ω⁻¹R, xtx, xty,
        storage_nd_1, storage_nd_2, storage_pd,
        storage_n_d, storage_n_p, storage_p_d,
        storage_d_d_1, storage_d_d_2, storage_d_d_3, 
        storage_d_d_4, storage_d_d_5, storage_d_d_6, storage_d_d_7,
        storage_p_p, storage_nd_nd, storage_pd_pd, logl,
        storages_nd_nd, Bcov, Σcov,
        P, invP, storage_nd_miss, storage_nd_nd_miss,
        Y_reml, X_reml, V_reml, B_reml, Ω_reml, R_reml,
        storage_nd_nd_reml, storage_pd_pd_reml, storage_n_p_reml,
        storage_nd_1_reml, storage_nd_2_reml, storage_n_d_reml,
        storage_p_d_reml, storage_pd_reml, logl_reml, Bcov_reml,
        se, reml)
end

const MultiResponseVarianceComponentModel = MRVCModel

MRVCModel(Y::AbstractMatrix, x::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRVCModel(Y, reshape(x, length(x), 1), V; kwargs...)

MRVCModel(y::AbstractVector, X::AbstractMatrix, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRVCModel(reshape(y, length(y), 1), X, V; kwargs...)

MRVCModel(y::AbstractVector, x::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRVCModel(reshape(y, length(y), 1), reshape(x, length(x), 1), V; kwargs...)

MRVCModel(Y::AbstractMatrix, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRVCModel(Y, nothing, V; kwargs...)

MRVCModel(y::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRVCModel(reshape(y, length(y), 1), nothing, V; kwargs...)

MRVCModel(Y, X, V::AbstractMatrix; kwargs...) = MRVCModel(Y, X, [V]; kwargs...)

MRVCModel(Y, V::AbstractMatrix; kwargs...) = MRVCModel(Y, [V]; kwargs...)

function Base.show(io::IO, model::MRVCModel)
    if model.reml
        n, d, p, m = size(model.Y_reml, 1), size(model.Y_reml, 2), size(model.X_reml, 2), length(model.V_reml)
    else
        n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.V)
    end
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
include("missing.jl")

end