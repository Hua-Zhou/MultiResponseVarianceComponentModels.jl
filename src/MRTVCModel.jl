
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
        B, Σ, Φ, Λ, logdetΣ2,
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
