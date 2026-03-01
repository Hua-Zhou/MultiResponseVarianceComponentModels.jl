"""
    MRVCModel{T, K, VarCompTypes}

Multivariate Response Variance Components Model.

Supports arbitrary number K of variance components with efficient dispatch
for common cases (especially K=2).

# Type Parameters
- `T`: Element type (Float64, Float32, etc.)
- `K`: Number of variance components (known at compile time)
- `VarCompTypes`: Tuple type of variance component structures

# Fields
- `Y`: Response matrix (n by d)
- `X`: Design matrix (n by p), or `nothing` for intercept-only
- `B`: Coefficient matrix (p by d)
- `Ω`: Lazy representation of the covariance structure, wrapping the variance components
- `se`: Whether to compute standard errors
"""

struct MRVCModel{T<:BlasReal,K,VarCompTypes<:Tuple{Vararg{VarCompStructure{T},K}}} <: VCModel
    # data
    Y::Matrix{T}
    X::Matrix{T}

    # parameters
    B::Matrix{T}
    Ω::ΩMatrix{T,K,VarCompTypes}

    # working arrays
    R::Matrix{T}                    # residuals
    Ωinv_R::Matrix{T}                 # Ωinv_R
    xtx::Matrix{T}                  # Gram matrix XᵀX
    xty::Matrix{T}                  # XᵀY
    storage_nd_1::Vector{T}
    storage_nd_2::Vector{T}
    storage_pd::Vector{T}
    storage_n_d::Matrix{T}
    storage_n_p::Matrix{T}
    storage_p_d::Matrix{T}
    storage_d_d_1::Matrix{T}
    storage_d_d_2::Matrix{T}
    storage_d_d_3::Matrix{T}
    storage_p_p::Matrix{T}
    storage_nd_nd::Matrix{T}
    storage_pd_pd::Matrix{T}
    storage_md²_md²::Matrix{T}
    logl::Vector{T}                 # log-likelihood

    # standard errors
    Bcov::Union{Nothing,Matrix{T}}
    Σcov::Union{Nothing,Matrix{T}}

    # options
    se::Bool
end

"""
    MRVCModel(Y, X, V, Σ; Σ_rank=..., se=true)

Construct variance component model from data and initial variance components.

# Arguments
- `Y::AbstractMatrix`: Response matrix (n by d)
- `X::AbstractMatrix`: Design matrix (n by p), with intercept column
- `V::Vector{Matrix}`: Fixed variance component matrices, length K (each n by n)
- `Σ::Vector{Matrix}`: Initial variance components, length K (each d by d)

# Keyword Arguments
- `Σ_rank::Vector{Int}`: Rank constraints for each component (default: full rank)
- `se::Bool`: Compute standard errors (default: true)

# Returns
- `MRVCModel{T, K, VarCompTypes}`: Fitted model structure
"""
function MRVCModel(
    Y::Union{AbstractMatrix{T},AbstractMatrix{Union{Missing,T}}},
    X::Union{Nothing,AbstractMatrix{T}},
    V::Vector{Matrix{T}},
    Σ::Vector{Matrix{T}};
    Σ_rank::Vector{Int}=fill(size(Y, 2), length(V)),
    se::Bool=true
) where {T<:BlasReal}
    # define dimensions
    K = length(V)
    n, d = size(Y)
    p = size(Xmat, 2)
    nd, pd = n * d, p * d

    (size(X, 1) == n && all(size.(V, 1) .== n)) || throw(DimensionMismatch())

    if X === nothing
        Xmat = Matrix{T}(undef, n, 0)
    else
        Xmat = X
    end

    length(V) == length(Σ) || throw(DimensionMismatch())

    # parameters -------------------------
    # Fixed Effects
    B = Matrix{T}(undef, p, d)

    # Construct variance components with appropriate structure
    function construct_VarComp(Σ::Matrix{T}, V::Matrix{T}, r::Int) where {T}
        n = checksquare(V)
        d = checksquare(Σ)
        if r < d
            return LowRankPlusDiagonal(Σ, r, V)
        else
            return Unstructured(Σ, V)
        end
    end

    VarComp = Tuple(construct_VarComp(Σ[i], V[i], Σ_rank[i]) for i in 1:K)

    # Create Ω representation
    Ω = ΩMatrix(VarComp)

    # working arrays
    R = Matrix{T}(undef, n, d)
    Ωinv_R = Matrix{T}(undef, n, d)
    xtx = transpose(Xmat) * Xmat
    xty = transpose(Xmat) * Y
    storage_nd_1 = Vector{T}(undef, nd)
    storage_nd_2 = Vector{T}(undef, nd)
    storage_pd = Vector{T}(undef, pd)
    storage_n_d = Matrix{T}(undef, n, d)
    storage_n_p = Matrix{T}(undef, n, p)
    storage_p_d = Matrix{T}(undef, p, d)
    storage_d_d_1 = Matrix{T}(undef, d, d)
    storage_d_d_2 = Matrix{T}(undef, d, d)
    storage_d_d_3 = Matrix{T}(undef, d, d)
    storage_p_p = Matrix{T}(undef, p, p)
    storage_nd_nd = Matrix{T}(undef, nd, nd)
    storage_pd_pd = Matrix{T}(undef, pd, pd)
    storage_md²_md² = Matrix{T}(undef, K * abs2(d), K * abs2(d))
    logl = ones(T, 1)

    if se == true
        Bcov = Matrix{T}(undef, pd, pd)
        dimΣcov = sum(x -> getfield(x, :pardim), VarComp)
        Σcov = Matrix{T}(undef, dimΣcov, dimΣcov)
    else
        Bcov = nothing
        Σcov = nothing
    end

    return MRVCModel{T,K,typeof(VarComp)}(
        Y,
        Xmat,
        B,
        VarComp,
        Ω,
        R,
        Ωinv_R,
        xtx,
        xty,
        storage_nd_1,
        storage_nd_2,
        storage_pd,
        storage_n_d,
        storage_n_p,
        storage_p_d,
        storage_d_d_1,
        storage_d_d_2,
        storage_d_d_3,
        storage_p_p,
        storage_nd_nd,
        storage_pd_pd,
        storage_md²_md²,
        logl,
        Bcov,
        Σcov,
        se,
    )
end

"""
    MRVCModel(Y, X, V; kwargs...)

Construct variance component model with automatic initialization of Σ to identity.

# Arguments
- `Y::AbstractMatrix`: Response matrix (n by d)
- `X::AbstractMatrix`: Design matrix (n by p)
- `V::Vector{Matrix}`: Fixed variance component matrices, length K (each n by n)

# Keyword Arguments
- `Σ_rank::Vector{Int}`: Rank constraints for each component (default: full rank)
- `se::Bool`: Compute standard errors (default: true)
"""
function MRVCModel(
    Y::Union{AbstractMatrix{T},AbstractMatrix{Union{Missing,T}}},
    X::Union{Nothing,AbstractMatrix{T}},
    V::Vector{Matrix{T}};
    kwargs...,
) where {T<:BlasReal}
    d = size(Y, 2)
    K = length(V)
    Σ_init = [Matrix{T}(I, d, d) for _ in 1:K]
    return MRVCModel(Y, X, V, Σ_init; kwargs...)
end

# Property accessors for backward compatibility
"""
    Base.getproperty(model::MRVCModel, name::Symbol)

Property accessor that provides backward-compatible Vector interface
for Σ and V, while internally using Tuple storage.
"""
function Base.getproperty(model::MRVCModel, name::Symbol)
    if name === :Σ
        # Return Vector of Σ matrices for backward compatibility
        return [vc.Σ for vc in getfield(model, :VarComp)]
    elseif name === :V
        # Return Vector of V matrices for backward compatibility
        return [vc.V for vc in getfield(model, :VarComp)]
    else
        return getfield(model, name)
    end
end

# Display method
function Base.show(io::IO, model::MRVCModel)
    n, d, p, m = size(model.Y, 1),
    size(model.Y, 2), size(model.X, 2),
    length(model.VarComp)

    if d == 1
        printstyled(
            io,
            "A univariate response variance component model\n";
            underline=true,
        )
    elseif d == 2
        printstyled(
            io,
            "A bivariate response variance component model\n";
            underline=true,
        )
    else
        printstyled(
            io,
            "A multivariate response variance component model\n";
            underline=true,
        )
    end

    print(io, "   * number of responses: ")
    printstyled(io, "$d\n"; color=:yellow)
    print(io, "   * number of observations: ")
    printstyled(io, "$n\n"; color=:yellow)
    print(io, "   * number of fixed effects: ")
    printstyled(io, "$p\n"; color=:yellow)
    print(io, "   * number of variance components: ")
    return printstyled(io, "$m"; color=:yellow)
end

# Convenience alias
const MultiResponseVarianceComponentModel = MRVCModel

# Utility functions
"""
    nvcomp(model::MRVCModel{T,K}) where {T,K}

Get number of variance components (compile-time constant).
"""
nvcomp(::MRVCModel{T,K}) where {T,K} = K

"""
    is_two(model::MRVCModel{T,K}) where {T,K}

Check if model has exactly two variance components (for dispatch).
"""
is_two(::MRVCModel{T,K}) where {T,K} = K == 2