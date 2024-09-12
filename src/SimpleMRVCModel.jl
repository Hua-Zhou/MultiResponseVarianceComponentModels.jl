struct SimpleMRVCModel{T<:BlasReal, K, VarCompTypes<:Tuple{Vararg{VarCompStructure{T},K}}} <: VCModel
    # data
    Y                       :: Matrix{T}
    X                       :: Matrix{T}
    # parameters
    B                       :: Matrix{T}
    VarComp                 :: VarCompTypes
    Ω                       :: Matrix{T} # covariance Ω = Σ[1]⊗V[1] + ... + Σ[m]⊗V[m]
    # working arrays 
    R                       :: Matrix{T} # residuals
    Ω⁻¹R                    :: Matrix{T}
    xtx                     :: Matrix{T} # Gram matrix XᵀX
    xty                     :: Matrix{T} # XᵀY
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
    Bcov                    :: Matrix{T} # for fisher_B!
    Σcov                    :: Matrix{T} # for fisher_Σ!
    # indicator for se, reml, missing response
    se                      :: Bool
    reml                    :: Bool
end

"""
    SimpleMRVCModel(
        Y::AbstractVecOrMat,
        X::Union{Nothing, AbstractVecOrMat},
        V::Union{AbstractMatrix, Vector{<:AbstractMatrix}}
        )

Create a new `SimpleMRVCModel` instance from response matrix `Y`, predictor matrix `X`, 
and kernel matrices `V`.

# Keyword arguments
```
se::Bool        calculate standard errors; default true
reml::Bool      pursue REML estimation instead of ML estimation; default false
```
"""
function SimpleMRVCModel(
    Y      :: Union{AbstractMatrix{T}, AbstractMatrix{Union{Missing, T}}},
    X      :: Union{Nothing, AbstractMatrix{T}},
    V      :: Vector{Matrix{T}},
    Σ      :: Vector{Matrix{T}};
    Σ_rank :: Vector{Int} = fill(size(Y, 2), length(V)),
    se     :: Bool = true
    ) where {T<:BlasReal}
    if X === nothing
        reml = false # REML = MLE in this case
        Xmat = Matrix{T}(undef, size(Y, 1), 0)
    else
        Xmat = X
    end
    # define dimensions
    n, d = size(Y)
    (size(Xmat, 1) == n && all(size.(V, 1) .== n)) || throw(DimensionMismatch())
    @assert length(V) == length(Σ)
    # `K` refers to the number of variance component matrices
    p, K = size(Xmat, 2), length(V)
    nd, pd = n * d, p * d
    # parameters
    B                = Matrix{T}(undef, p, d)
    # TODO: add a method for selecting variance components to be Unstructured vs Structured
    # For now, Σranks as selector
    function construct_VarComp(Σ::Matrix{T}, V::Matrix{T}, r::Int) where {T}
        n = LinearAlgebra.checksquare(V)
        d = LinearAlgebra.checksquare(Σ)
        if r < d
            return LowRankPlusDiagonal(Σ, r, V)
        else
            return Unstructured(Σ, V)
        end
    end
    VarComp          = Tuple(construct_VarComp(Σ[i], V[i], Σ_rank[i]) for i in 1:K)
    Ω                = Matrix{T}(undef, nd, nd)

    # working arrays
    R                = Matrix{T}(undef, n, d)
    Ω⁻¹R             = Matrix{T}(undef, n, d)
    xtx              = transpose(Xmat) * Xmat
    xty              = transpose(Xmat) * Y
    storage_nd_1     = Vector{T}(undef, nd)
    storage_nd_2     = Vector{T}(undef, nd)
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
    logl             = ones(T,1)
    
    if se == true
        Bcov           = Matrix{T}(undef, pd, pd)
        # TODO: Σcov has to be initialized after VarCompStructures are known
        dimΣvcov       = sum(x -> getfield(x,:pardim), VarComp)
        Σcov           = Matrix{T}(undef, dimΣvcov, dimΣvcov)
    else
        Bcov = nothing
        Σcov = nothing
    end

    SimpleMRVCModel{T,K,typeof(VarComp)}(
        Y, Xmat,
        B, VarComp, Ω,
        R, Ω⁻¹R, xtx, xty,
        storage_nd_1, storage_nd_2, storage_pd,
        storage_n_d, storage_n_p, storage_p_d,
        storage_d_d_1, storage_d_d_2, storage_d_d_3, 
        storage_d_d_4, storage_d_d_5, storage_d_d_6, storage_d_d_7,
        storage_p_p, storage_nd_nd, storage_pd_pd, logl,
        Bcov, Σcov,
        se, false
        )
end

function Base.show(io::IO, model::SimpleMRVCModel)
    if model.reml
        n, d, p, m = size(model.Y_reml, 1), size(model.Y_reml, 2), size(model.X_reml, 2), length(model.V_reml)
    else
        n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.VarComp)
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

# const SimpleMultiResponseVarianceComponentModel = SimpleMRVCModel

# SimpleMRVCModel(Y::AbstractMatrix, x::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
#     SimpleMRVCModel(Y, reshape(x, length(x), 1), V; kwargs...)

# SimpleMRVCModel(y::AbstractVector, X::AbstractMatrix, V::Vector{<:AbstractMatrix}; kwargs...) = 
#     SimpleMRVCModel(reshape(y, length(y), 1), X, V; kwargs...)

# SimpleMRVCModel(y::AbstractVector, x::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
#     SimpleMRVCModel(reshape(y, length(y), 1), reshape(x, length(x), 1), V; kwargs...)

# SimpleMRVCModel(Y::AbstractMatrix, V::Vector{<:AbstractMatrix}; kwargs...) = 
#     SimpleMRVCModel(Y, nothing, V; kwargs...)

# SimpleMRVCModel(y::AbstractVector, V::Vector{<:AbstractMatrix}; kwargs...) = 
#     SimpleMRVCModel(reshape(y, length(y), 1), nothing, V; kwargs...)

# SimpleMRVCModel(Y, X, V::AbstractMatrix; kwargs...) = SimpleMRVCModel(Y, X, [V]; kwargs...)

# SimpleMRVCModel(Y, V::AbstractMatrix; kwargs...) = SimpleMRVCModel(Y, [V]; kwargs...)

