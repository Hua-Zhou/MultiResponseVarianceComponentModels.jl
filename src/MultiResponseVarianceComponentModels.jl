"""
__MRVCModels__ stands for __M__ultivariate __R__esponse __V__ariance __C__omponents
linear mixed __Models__. `MRVCModels.jl` permits maximum likelihood (ML) or residual
maximum likelihood (REML) estimation and inference.
"""
module MultiResponseVarianceComponentModels

using IterativeSolvers, LinearAlgebra, Manopt, Manifolds, Distributions, SweepOperator, InvertedIndices
import LinearAlgebra: BlasReal, copytri!
export VCModel,
    MultiResponseVarianceComponentModel,
    MRVCModel,
    MRTVCModel,
    # fit.jl or eigen.jl
    fit!,
    loglikelihood!,
    loglikelihood,
    loglikelihood_reml,
    update_res!,
    update_Ω!,
    update_B!,
    update_B_reml!,
    fisher_B!,
    fisher_B_reml!,
    fisher_Σ!,
    # missing.jl
    permute,
    # parse.jl
    lrt,
    h2,
    rg,
    # mvcalculus.jl
    kron_axpy!,
    kron_reduction!,
    vech,
    ◺,
    duplication,
    commutation

abstract type VCModel end

struct MRVCModel{T <: BlasReal} <: VCModel
    # data
    Y                       :: Matrix{T}
    X                       :: Matrix{T}
    V                       :: Vector{Matrix{T}}
    # parameters
    B                       :: Matrix{T}
    Σ                       :: Vector{Matrix{T}}
    Ω                       :: Matrix{T} # covariance Ω = Σ[1]⊗V[1] + ... + Σ[m]⊗V[m]
    Γ                       :: Vector{Matrix{T}} # for manopt.jl
    Ψ                       :: Matrix{T}         # for manopt.jl
    Σ_rank                  :: Vector{Int}       # for manopt.jl
    # working arrays
    V_rank                  :: Vector{Int}
    R                       :: Matrix{T} # residuals
    Ω⁻¹R                    :: Matrix{T}
    xtx                     :: Matrix{T} # Gram matrix X'X
    xty                     :: Matrix{T} # X'Y
    storage_nd_1            :: Vector{T}
    storage_nd_2            :: Vector{T}
    storage_pd              :: Vector{T}
    storage_n_d             :: Matrix{T}
    storage_n_p             :: Matrix{T}
    storage_p_d             :: Matrix{T}
    storage_d_d_1           :: Matrix{T}
    storage_d_d_2           :: Matrix{T}
    storage_d_d_3           :: Matrix{T}
    storage_d_d_4           :: Matrix{T} # for manopt.jl
    storage_d_d_5           :: Matrix{T} # for manopt.jl
    storage_d_d_6           :: Matrix{T} # for manopt.jl
    storage_d_d_7           :: Matrix{T} # for manopt.jl
    storage_p_p             :: Matrix{T}
    storage_nd_nd           :: Matrix{T}
    storage_pd_pd           :: Matrix{T}
    logl                    :: Vector{T} # log-likelihood
    # standard errors
    storages_nd_nd          :: Union{Nothing, Vector{Matrix{T}}} # for fisher_Σ!
    Bcov                    :: Union{Nothing, Matrix{T}} # for fisher_B!
    Σcov                    :: Union{Nothing, Matrix{T}} # for fisher_Σ!
    # permutation for missing response
    P                       :: Union{Nothing, Vector{Int}}
    invP                    :: Union{Nothing, Vector{Int}}
    n_miss                  :: Int
    Y_obs                   :: Union{Nothing, Vector{T}}
    # working arrays for missing response
    storage_n_miss_n_obs_1  :: Union{Nothing, Matrix{T}} # for imputed response
    storage_n_miss_n_obs_2  :: Union{Nothing, Matrix{T}}
    storage_n_miss_n_obs_3  :: Union{Nothing, Matrix{T}}
    storage_n_miss_n_miss_1 :: Union{Nothing, Matrix{T}} # conditional variance
    storage_n_miss_n_miss_2 :: Union{Nothing, Matrix{T}}
    storage_n_miss_n_miss_3 :: Union{Nothing, Matrix{T}} 
    storage_nd_nd_miss      :: Union{Nothing, Matrix{T}}
    storage_d_d_miss        :: Union{Nothing, Matrix{T}}
    storage_n_obs           :: Union{Nothing, Vector{T}}
    storage_n_miss          :: Union{Nothing, Vector{T}}
    # original data for reml
    Y_reml                  :: Union{Nothing, Matrix{T}}
    X_reml                  :: Union{Nothing, Matrix{T}}
    V_reml                  :: Union{Nothing, Vector{Matrix{T}}}
    # fixed effects parameters for reml
    B_reml                  :: Union{Nothing, Matrix{T}}
    # working arrays for reml
    Ω_reml                  :: Union{Nothing, Matrix{T}}
    R_reml                  :: Union{Nothing, Matrix{T}}
    storage_nd_nd_reml      :: Union{Nothing, Matrix{T}}
    storage_pd_pd_reml      :: Union{Nothing, Matrix{T}}
    storage_n_p_reml        :: Union{Nothing, Matrix{T}}
    storage_nd_1_reml       :: Union{Nothing, Vector{T}}
    storage_nd_2_reml       :: Union{Nothing, Vector{T}}
    storage_n_d_reml        :: Union{Nothing, Matrix{T}}
    storage_p_d_reml        :: Union{Nothing, Matrix{T}}
    storage_pd_reml         :: Union{Nothing, Vector{T}}
    logl_reml               :: Union{Nothing, Vector{T}}
    Bcov_reml               :: Union{Nothing, Matrix{T}}
    # indicator for se, reml, missing response
    se                      :: Bool
    reml                    :: Bool
    ymissing                :: Bool
end

"""
    MRVCModel(
        Y::AbstractVecOrMat,
        X::Union{Nothing, AbstractVecOrMat},
        V::Union{AbstractMatrix, Vector{<:AbstractMatrix}}
        )

Create a new `MRVCModel` instance from response matrix `Y`, predictor matrix `X`, 
and kernel matrices `V`.

# Keyword arguments
```
se::Bool        calculate standard errors; default true
reml::Bool      pursue REML estimation instead of ML estimation; default false
```

# Extended help
When there are two variance components, computation can be reduced by avoiding large matrix 
inversion per iteration, which is achieved with `MRTVCModel` instance. __MRTVCModels__ 
stands for __M__ultivariate __R__esponse __T__wo __V__ariance __C__omponents
linear mixed __Models__. `MRVCModel` is more general and is not limited to two variance 
components case. For `MRTVCModel`, the number of variance components must be two.
"""
function MRVCModel(
    Y      :: Union{AbstractMatrix{T}, AbstractMatrix{Union{Missing, T}}},
    X      :: Union{Nothing, AbstractMatrix{T}},
    V      :: Vector{<:AbstractMatrix{T}};
    Σ_rank :: Vector{<:Integer} = fill(size(Y, 2), length(V)),
    se     :: Bool = true,
    reml   :: Bool = false
    ) where T <: BlasReal
    if X === nothing
        reml = false # REML = MLE in this case
        Xmat = Matrix{T}(undef, size(Y, 1), 0)
    else
        Xmat = X
    end
    @assert size(Y, 1) == size(Xmat, 1) == size(V[1], 1)
    # define dimesions
    n, p, d, m = size(Y, 1), size(Xmat, 2), size(Y, 2), length(V)
    nd, pd = n * d, p * d
    storage_nd_1 = Vector{T}(undef, nd)
    storage_nd_2 = Vector{T}(undef, nd)
    if any(ismissing, Y)
        @assert reml == false "only ML estimation is possible for missing response"
        @assert se == false "standard errors cannot be computed for missing response"
        P, invP, n_miss, Y = permute(Y)
        ymissing = true
        n_obs = nd - n_miss
        copyto!(storage_nd_1, Y)
        storage_nd_2 .= @view storage_nd_1[P]
        Y_obs = storage_nd_2[1:n_obs]
        storage_n_miss_n_obs_1  = Matrix{T}(undef, n_miss, n_obs)
        storage_n_miss_n_obs_2  = Matrix{T}(undef, n_miss, n_obs)
        storage_n_miss_n_obs_3  = Matrix{T}(undef, n_miss, n_obs)
        storage_n_miss_n_miss_1 = Matrix{T}(undef, n_miss, n_miss)
        storage_n_miss_n_miss_2 = Matrix{T}(undef, n_miss, n_miss)
        storage_n_miss_n_miss_3 = Matrix{T}(undef, n_miss, n_miss)
        storage_nd_nd_miss      = Matrix{T}(undef, nd, nd)
        storage_d_d_miss        = Matrix{T}(undef, d, d)
        storage_n_obs           = Vector{T}(undef, n_obs)
        storage_n_miss          = Vector{T}(undef, n_miss)
    else
        P = invP = Y_obs = storage_n_miss_n_obs_1 = storage_n_miss_n_obs_2 = 
            storage_n_miss_n_obs_3 = storage_n_miss_n_miss_1 = storage_n_miss_n_miss_2 =
            storage_n_miss_n_miss_3 = storage_nd_nd_miss = storage_d_d_miss = 
            storage_n_obs = storage_n_miss = nothing
        n_miss = 0
        ymissing = false
    end
    if reml
        Y_reml  = deepcopy(Y)
        X_reml  = deepcopy(Xmat)
        V_reml  = deepcopy(V)
        Y, V, _ = project_null(Y_reml, X_reml, V_reml)
        Xmat    = Matrix{T}(undef, size(Y, 1), 0)
        # re-define dimensions
        n_reml, p_reml = n, p
        n, p = size(Y, 1), 0
        nd, pd = n * d, p * d
        nd_reml, pd_reml = n_reml * d, p_reml * d
        B_reml             = Matrix{T}(undef, p_reml, d)
        Ω_reml             = Matrix{T}(undef, nd_reml, nd_reml)
        R_reml             = Matrix{T}(undef, n_reml, d)
        storage_nd_nd_reml = Matrix{T}(undef, nd_reml, nd_reml)
        storage_pd_pd_reml = Matrix{T}(undef, pd_reml, pd_reml)
        storage_n_p_reml   = Matrix{T}(undef, n_reml, p_reml)
        storage_nd_1_reml  = Vector{T}(undef, nd_reml)
        storage_nd_2_reml  = Vector{T}(undef, nd_reml)
        storage_n_d_reml   = Matrix{T}(undef, n_reml, d)
        storage_p_d_reml   = Matrix{T}(undef, p_reml, d)
        storage_pd_reml    = Vector{T}(undef, pd_reml)
        logl_reml          = zeros(T, 1)
    else
        Y_reml = X_reml = V_reml = B_reml = Ω_reml = R_reml = 
            storage_nd_nd_reml = storage_pd_pd_reml = 
            storage_n_p_reml = storage_nd_1_reml = 
            storage_nd_2_reml = storage_n_d_reml = 
            storage_p_d_reml = storage_pd_reml = 
            logl_reml = Bcov_reml = nothing
    end
    if se
        storages_nd_nd = [Matrix{T}(undef, nd, nd) for _ in 1:m]
        Bcov           = Matrix{T}(undef, pd, pd)
        Σcov           = Matrix{T}(undef, m * ◺(d), m * ◺(d))
        reml ? Bcov_reml  = Matrix{T}(undef, pd_reml, pd_reml) : Bcov_reml  = nothing
    else
        storages_nd_nd = Bcov = Σcov = Bcov_reml = nothing
    end
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
        P, invP, n_miss, Y_obs,
        storage_n_miss_n_obs_1, storage_n_miss_n_obs_2,
        storage_n_miss_n_obs_3, storage_n_miss_n_miss_1,
        storage_n_miss_n_miss_2, storage_n_miss_n_miss_3,
        storage_nd_nd_miss, storage_d_d_miss,
        storage_n_obs, storage_n_miss,
        Y_reml, X_reml, V_reml, B_reml, Ω_reml, R_reml,
        storage_nd_nd_reml, storage_pd_pd_reml, storage_n_p_reml,
        storage_nd_1_reml, storage_nd_2_reml, storage_n_d_reml,
        storage_p_d_reml, storage_pd_reml, logl_reml, Bcov_reml,
        se, reml, ymissing
        )
end

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

const MultiResponseVarianceComponentModel = MRVCModel

struct MRTVCModel{T <: BlasReal} <: VCModel
    # data
    Y                       :: Matrix{T}
    Ỹ                       :: Matrix{T}
    X                       :: Matrix{T}
    X̃                       :: Matrix{T}
    V                       :: Vector{Matrix{T}}
    U                       :: Matrix{T}
    D                       :: Vector{T}
    logdetV2                :: T
    # parameters
    B                       :: Matrix{T}
    Σ                       :: Vector{Matrix{T}}
    Φ                       :: Matrix{T}
    Λ                       :: Vector{T}
    logdetΣ2                :: Vector{T}
    V_rank                  :: Vector{Int}
    # working arrays
    xtx                     :: Matrix{T} # Gram matrix X'X
    xty                     :: Matrix{T} # X'Y
    ỸΦ                      :: Matrix{T}
    R̃                       :: Matrix{T}
    R̃Φ                      :: Matrix{T}
    N1tN1                   :: Matrix{T}
    N2tN2                   :: Matrix{T}
    storage_d_1             :: Vector{T}
    storage_d_2             :: Vector{T}
    storage_d_d_1           :: Matrix{T}
    storage_d_d_2           :: Matrix{T}
    storage_p_p             :: Matrix{T}
    storage_pd              :: Vector{T}
    storage_pd_pd           :: Matrix{T}
    storage_nd_1            :: Vector{T}
    storage_nd_2            :: Vector{T}
    storage_nd_pd           :: Matrix{T}
    logl                    :: Vector{T} # log-likelihood
    # standard errors
    Bcov                    :: Union{Nothing, Matrix{T}} # for fisher_B!
    Σcov                    :: Union{Nothing, Matrix{T}} # for fisher_Σ!
    # original data for reml
    Y_reml                  :: Union{Nothing, Matrix{T}}
    Ỹ_reml                  :: Union{Nothing, Matrix{T}}
    X_reml                  :: Union{Nothing, Matrix{T}}
    X̃_reml                  :: Union{Nothing, Matrix{T}}
    V_reml                  :: Union{Nothing, Vector{Matrix{T}}}
    U_reml                  :: Union{Nothing, Matrix{T}}
    D_reml                  :: Union{Nothing, Vector{T}}
    logdetV2_reml           :: Union{Nothing, T}
    # fixed effects parameters for reml
    B_reml                  :: Union{Nothing, Matrix{T}}
    # working arrays for reml
    ỸΦ_reml                 :: Union{Nothing, Matrix{T}}
    R̃_reml                  :: Union{Nothing, Matrix{T}}
    R̃Φ_reml                 :: Union{Nothing, Matrix{T}}
    storage_nd_pd_reml      :: Union{Nothing, Matrix{T}}
    storage_nd_1_reml       :: Union{Nothing, Vector{T}}
    storage_nd_2_reml       :: Union{Nothing, Vector{T}}
    storage_pd_pd_reml      :: Union{Nothing, Matrix{T}}
    storage_pd_reml         :: Union{Nothing, Vector{T}}
    logl_reml               :: Union{Nothing, Vector{T}}
    Bcov_reml               :: Union{Nothing, Matrix{T}}
    # indicator for se, reml
    se                      :: Bool
    reml                    :: Bool
end

"""
    MRTVCModel(
        Y::AbstractVecOrMat,
        X::Union{Nothing, AbstractVecOrMat},
        V::Vector{<:AbstractMatrix}
        )

Create a new `MRTVCModel` instance from response matrix `Y`, predictor matrix `X`, 
and kernel matrices `V`. The number of variance components must be two.

# Keyword arguments
```
se::Bool        calculate standard errors; default true
reml::Bool      pursue REML estimation instead of ML estimation; default false
```
"""
function MRTVCModel(
    Y      :: AbstractMatrix{T},
    X      :: Union{Nothing, AbstractMatrix{T}},
    V      :: Vector{<:AbstractMatrix{T}};
    se     :: Bool = true,
    reml   :: Bool = false
    ) where T <: BlasReal
    if X === nothing
        reml = false # REML = MLE in this case
        Xmat = Matrix{T}(undef, size(Y, 1), 0)
    else
        Xmat = X
    end
    @assert length(V) == 2
    @assert size(Y, 1) == size(Xmat, 1) == size(V[1], 1)
    # define dimesions
    n, p, d, m = size(Y, 1), size(Xmat, 2), size(Y, 2), 2
    nd, pd = n * d, p * d
    if reml
        Y_reml  = deepcopy(Y)
        X_reml  = deepcopy(Xmat)
        V_reml  = deepcopy(V)
        Y, V, _ = project_null(Y_reml, X_reml, V_reml)
        Xmat    = Matrix{T}(undef, size(Y, 1), 0)
        # re-define dimensions
        n_reml, p_reml = n, p
        n, p = size(Y, 1), 0
        nd, pd = n * d, p * d
        nd_reml, pd_reml = n_reml * d, p_reml * d
        D_reml, U_reml = eigen(Symmetric(V_reml[1]), Symmetric(V_reml[2]))
        logdetV2_reml = logdet(V_reml[2])
        Ỹ_reml = transpose(U_reml) * Y_reml
        X̃_reml = transpose(U_reml) * X_reml
        B_reml             = Matrix{T}(undef, p_reml, d)
        ỸΦ_reml            = Matrix{T}(undef, n_reml, d)
        R̃_reml             = Matrix{T}(undef, n_reml, d)
        R̃Φ_reml            = Matrix{T}(undef, n_reml, d)
        storage_nd_pd_reml = Matrix{T}(undef, nd_reml, pd_reml)
        storage_nd_1_reml  = Vector{T}(undef, nd_reml)
        storage_nd_2_reml  = Vector{T}(undef, nd_reml)
        storage_pd_pd_reml = Matrix{T}(undef, pd_reml, pd_reml)
        storage_pd_reml    = Vector{T}(undef, pd_reml)
        logl_reml          = zeros(T, 1)    
    else
        Y_reml = Ỹ_reml = X_reml = X̃_reml = V_reml = U_reml = D_reml =
            logdetV2_reml = B_reml = ỸΦ_reml = R̃_reml = R̃Φ_reml =
            storage_nd_pd_reml = storage_nd_1_reml = 
            storage_nd_2_reml = storage_pd_pd_reml = storage_pd_reml = 
            logl_reml = Bcov_reml = nothing        
    end
    if se
        Bcov           = Matrix{T}(undef, pd, pd)
        Σcov           = Matrix{T}(undef, m * ◺(d), m * ◺(d))
        reml ? Bcov_reml  = Matrix{T}(undef, pd_reml, pd_reml) : Bcov_reml  = nothing
    else
        Bcov = Σcov = Bcov_reml = nothing
    end
    D, U = eigen(Symmetric(V[1]), Symmetric(V[2]))
    logdetV2 = logdet(V[2])
    Ỹ = transpose(U) * Y
    X̃ = p == 0 ? Matrix{T}(undef, n, 0) : transpose(U) * Xmat
    # parameters
    B                = Matrix{T}(undef, p, d)
    Σ                = [Matrix{T}(undef, d, d) for _ in 1:m]
    Φ                = Matrix{T}(undef, d, d)
    Λ                = Vector{T}(undef, d)
    logdetΣ2         = zeros(T, 1)
    V_rank           = [rank(V[k]) for k in 1:m]
    # working arrays
    xtx              = transpose(Xmat) * Xmat
    xty              = transpose(Xmat) * Y
    ỸΦ               = Matrix{T}(undef, n, d)
    R̃                = Matrix{T}(undef, n, d)
    R̃Φ               = Matrix{T}(undef, n, d)
    N1tN1            = Matrix{T}(undef, d, d)
    N2tN2            = Matrix{T}(undef, d, d)
    storage_d_1      = Vector{T}(undef, d)
    storage_d_2      = Vector{T}(undef, d)
    storage_d_d_1    = Matrix{T}(undef, d, d)
    storage_d_d_2    = Matrix{T}(undef, d, d)
    storage_p_p      = Matrix{T}(undef, p, p)
    storage_pd       = Vector{T}(undef, pd)
    storage_pd_pd    = Matrix{T}(undef, pd, pd)
    storage_nd_1     = Vector{T}(undef, nd)
    storage_nd_2     = Vector{T}(undef, nd)
    storage_nd_pd    = Matrix{T}(undef, nd, pd)
    logl             = zeros(T, 1)
    MRTVCModel{T}(
        Y, Ỹ, Xmat, X̃, V, U, D, logdetV2,
        B, Σ, Φ, Λ, logdetΣ2, V_rank,
        xtx, xty, ỸΦ, R̃, R̃Φ, N1tN1, N2tN2,
        storage_d_1, storage_d_2, storage_d_d_1, storage_d_d_2,
        storage_p_p, storage_pd, storage_pd_pd, 
        storage_nd_1, storage_nd_2, storage_nd_pd, logl, Bcov, Σcov,
        Y_reml, Ỹ_reml, X_reml, X̃_reml, V_reml, U_reml, D_reml,
        logdetV2_reml, B_reml, ỸΦ_reml, R̃_reml, R̃Φ_reml,
        storage_nd_pd_reml, storage_nd_1_reml, 
        storage_nd_2_reml, storage_pd_pd_reml, storage_pd_reml,
        logl_reml, Bcov_reml, se, reml
        )
end

MRTVCModel(Y::AbstractMatrix, x::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRTVCModel(Y, reshape(x, length(x), 1), V; kwargs...)

MRTVCModel(y::AbstractVector, X::AbstractMatrix, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRTVCModel(reshape(y, length(y), 1), X, V; kwargs...)

MRTVCModel(y::AbstractVector, x::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRTVCModel(reshape(y, length(y), 1), reshape(x, length(x), 1), V; kwargs...)

MRTVCModel(Y::AbstractMatrix, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRTVCModel(Y, nothing, V; kwargs...)

MRTVCModel(y::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
    MRTVCModel(reshape(y, length(y), 1), nothing, V; kwargs...)

function Base.show(io::IO, model::VCModel)
    if model.reml
        n, d, p, m = size(model.Y_reml, 1), size(model.Y_reml, 2), size(model.X_reml, 2), length(model.V_reml)
    else
        n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.V)
    end
    if d == 1 && model isa MRTVCModel
        printstyled(io, "A univariate response two variance component model\n"; underline = true)
    elseif d == 1
        printstyled(io, "A univariate response variance component model\n"; underline = true)
    elseif d == 2 && model isa MRTVCModel
        printstyled(io, "A bivariate response two variance component model\n"; underline = true)
    elseif d == 2
        printstyled(io, "A bivariate response variance component model\n"; underline = true)
    elseif model isa MRTVCModel
        printstyled(io, "A multivariate response two variance component model\n"; underline = true)
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

include("mvcalculus.jl")
include("reml.jl")
include("fit.jl")
include("eigen.jl")
include("manopt.jl")
include("parse.jl")
include("missing.jl")

end