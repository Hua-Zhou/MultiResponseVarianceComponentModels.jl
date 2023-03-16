struct MultiResponseVarianceComponentModel{T <: BlasReal}
    # data
    Y                :: Matrix{T}
    X                :: Matrix{T}
    V                :: Vector{Matrix{T}}
    # parameters
    B                :: Matrix{T}
    VarComp          :: Vector{VarCompStructure{T}}
    Ω                :: Matrix{T} # covariance Ω = Σ[1] ⊗ V[1] + ... + Σ[m] ⊗ V[m]
    Σ_rank           :: Vector{Int}
    # working arrays
    RtVR             :: Vector{Matrix{T}}
    storages_d_d     :: Vector{Matrix{T}}
    V_sqnorm         :: Matrix{T}
    V_rank           :: Vector{Int}
    R                :: Matrix{T} # residuals
    Ω⁻¹R             :: Matrix{T}
    xtx              :: Matrix{T} # Gram matrix X'X
    xty              :: Matrix{T} # X'Y
    storage_d        :: Vector{T}
    storage_nd_1     :: Vector{T}
    storage_nd_2     :: Vector{T}
    storage_nd_3     :: Vector{T}
    storage_pd       :: Vector{T}
    storage_n_d_1    :: Matrix{T}
    storage_n_d_2    :: Matrix{T}
    storage_n_d_3    :: Matrix{T}
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
    storages_nd_nd   :: Vector{Matrix{T}}
    Bcov             :: Matrix{T}
    Σcov             :: Matrix{T}
    logl             :: Vector{T}
end

# constructor
function MultiResponseVarianceComponentModel(
    Y      :: AbstractMatrix{T},
    X      :: Union{Nothing, Matrix{T}},
    V      :: Vector{<:AbstractMatrix{T}};
    Σ_rank :: Vector{<:Integer} = fill(size(Y, 2), length(V))
    ) where T <: BlasReal
    # dimensions
    n, d, m = size(Y, 1), size(Y, 2), length(V)
    if X === nothing
        p = 0
        Xmat = Matrix{T}(undef, n, 0)
    else
        p = size(X, 2)
        Xmat = X
    end
    nd, pd = n * d, p * d
    # parameters
    B                = Matrix{T}(undef, p, d)
    VarComp          = Vector{VarCompStructure{T}}(undef, m)
    for i in 1:m
        if Σ_rank[i] < d
            VarComp[i] = LowRankPlusDiagonal(d, Σ_rank[i], T)
        else
            VarComp[i] = Unstructured(d, T)
        end
    end
    Ω                = Matrix{T}(undef, nd, nd)
    RtVR             = [Matrix{T}(undef, d, d) for _ in 1:m]
    storages_d_d     = [Matrix{T}(undef, d, d) for _ in 1:m]
    V_sqnorm         = Matrix{T}(undef, m, m)
    for j in 1:m
        for i in j:m
            V_sqnorm[i, j] = dot(V[i], V[j])
        end
    end
    copytri!(V_sqnorm, 'L')
     # TODO: estimates rank by svd, do we need this?
    V_rank           = [rank(V[k]) for k in 1:m]
    # working arrays
    R                = Matrix{T}(undef, n, d)
    Ω⁻¹R             = Matrix{T}(undef, n, d)
    xtx              = transpose(Xmat) * Xmat
    xty              = transpose(Xmat) * Y
    storage_d        = Vector{T}(undef, d)
    storage_nd_1     = Vector{T}(undef, nd)
    storage_nd_2     = Vector{T}(undef, nd)
    storage_nd_3     = Vector{T}(undef, nd)
    storage_pd       = Vector{T}(undef, pd)
    storage_n_d_1    = Matrix{T}(undef, n, d)
    storage_n_d_2    = Matrix{T}(undef, n, d)
    storage_n_d_3    = Matrix{T}(undef, n, d)
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
        B, VarComp, Ω, Σ_rank, 
        RtVR, storages_d_d, V_sqnorm, V_rank,
        R, Ω⁻¹R, xtx, xty,
        storage_d, storage_nd_1, storage_nd_2, storage_nd_3, storage_pd,
        storage_n_d_1, storage_n_d_2, storage_n_d_3,
        storage_n_p, storage_p_d,
        storage_d_d_1, storage_d_d_2, storage_d_d_3, 
        storage_d_d_4, storage_d_d_5, storage_d_d_6, storage_d_d_7,
        storage_p_p, storage_nd_nd, storage_pd_pd,
        storages_nd_nd, Bcov, Σcov, logl)
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

# Methods
function Base.show(io::IO, model::MultiResponseVarianceComponentModel)
    n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.V)
    if d == 1
        print(io, "A univariate response model (n = $n, p = $p, m = $m)")
    else
        print(io, "A multivariate response model (n = $n, d = $d, p = $p, m = $m)")
    end
end

"""
    fit!(model::MultiResponseVarianceComponentModel)

Fit a multivariate response variance component model by an MM or EM algorithm.

# Positional arguments
- `model`            : a `MultiResponseVarianceComponentModel` instance.  

# Keyword arguments
- `maxiter::Integer` : maximum number of iterations. Default is `1000`.
- `reltol::Real`     : relative tolerance for convergence. Default is `1e-6`.
- `verbose::Bool`    : display algorithmic information. Default is `false`.
- `init::Symbol`     : initialization strategy. `:default` initialize by least squares.
    `:user` uses user supplied value at `model.B` and `model.Σ`.
- `algo::Symbol`     : optimization algorithm. `:MM` (default) or `EM`.
- `log::Bool`        : record iterate history or not. Defaut is `false`.
- `reml::Bool`       : REML instead of ML estimation. Default is `false`.
- `se::Bool`         : calculate standard errors. Default is `true`.

# Output
- `history`          : iterate history.
"""
function fit!(
    model   :: MultiResponseVarianceComponentModel{T};
    maxiter :: Integer = 1000,
    reltol  :: Real = 1e-6,
    verbose :: Bool = false,
    init    :: Symbol = :default, # :default or :user
    algo    :: Symbol = :MM,
    log     :: Bool = false,
    reml    :: Bool = false,
    se      :: Bool = true
    ) where T <: BlasReal
    Y, X, V = model.Y, model.X, model.V
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(V)
    @info "n = $n, d = $d, p = $p, m = $m"
    if reml
        Ỹ, Ṽ, _ = project_null(model.Y, model.X, model.V)
        modelf = MultiResponseVarianceComponentModel(Ỹ, Ṽ)
        @info "running $(algo) algorithm for REML estimation"
    else
        modelf = model
        @info "running $(algo) algorithm for ML estimation"
    end
    # record iterate history if requested
    history          = ConvergenceHistory(partial = !log)
    history[:reltol] = reltol
    IterativeSolvers.reserve!(Int    , history, :iter    , maxiter + 1)
    IterativeSolvers.reserve!(T      , history, :logl    , maxiter + 1)
    IterativeSolvers.reserve!(Float64, history, :itertime, maxiter + 1)
    # initialization
    tic = time()
    if init == :default
        initialize!(modelf)
    elseif init == :user
        if p > 0 
            update_res!(modelf)
        else
            copy!(modelf.R, Y)
        end
        update_Ω!(modelf)
    else
        throw("unrecognize initialization method $init")
    end
    logl = loglikelihood!(modelf)
    toc = time()
    verbose && println("iter = 0, logl = $logl")
    IterativeSolvers.nextiter!(history)
    push!(history, :iter    , 0)
    push!(history, :logl    , logl)
    push!(history, :itertime, toc - tic)
    # MM loop
    for iter in 1:maxiter
        IterativeSolvers.nextiter!(history)
        tic = time()
        # initial estimate of Σ can be lousy, so we update Σ first in the 1st round
        p > 0 && iter > 1 && update_B!(modelf)
        update_Σ!(modelf, algo = algo)
        logl_prev = logl
        logl = loglikelihood!(modelf)
        toc = time()
        verbose && println("iter = $iter, logl = $logl")
        push!(history, :iter    , iter)
        push!(history, :logl    , logl)
        push!(history, :itertime, toc - tic)
        if iter == maxiter
            @warn "maximum number of iterations $maxiter is reached."
            break
        end
        if abs(logl - logl_prev) < reltol * (abs(logl_prev) + 1)
            @info "updates converged!"
            copyto!(modelf.logl, logl)
            IterativeSolvers.setconv(history, true)
            if se
                @info "calculating standard errors"
                fisher_B!(modelf)
                fisher_Σ!(modelf)
            end
            break
        end
    end
    if reml == true
        copy!(model.Σ, modelf.Σ)
        copyto!(model.Σcov, modelf.Σcov)
        update_Ω!(model)
        # need Ω⁻¹ for B 
        copyto!(model.storage_nd_nd, model.Ω)
        # Cholesky of covariance Ω = U'U
        _, info = LAPACK.potrf!('U', model.storage_nd_nd)
        info > 0 && throw("covariance matrix Ω is singular")
        LAPACK.potri!('U', model.storage_nd_nd)
        copytri!(model.storage_nd_nd, 'U')
        update_B!(model)
        copyto!(model.logl, loglikelihood!(model))
        se ? fisher_B!(modelf) : nothing
    end
    log && IterativeSolvers.shrink!(history)
    return history
end

"""
    initialize!(model::MultiResponseVarianceComponentModel)

"""
function initialize!(
    model::MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    Y, X, V = model.Y, model.X, model.V
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(V)
    # Fixed Effects
    if p > 0
        # estimate B by ordinary least squares (Cholesky solve)
        copyto!(model.storage_p_p, model.xtx)
        _, info = LAPACK.potrf!('U', model.storage_p_p)
        info > 0 && throw("design matrix X is rank deficient")
        LAPACK.potrs!('U', model.storage_p_p, copyto!(model.B, model.xty))
        # update residuals R
        update_res!(model)
    else
        # no fixed effects
        copy!(model.R, Y)
    end
    # initialization of variance components Σ[1], ..., Σ[m]
    # vec(R) ∼ Normal(0, ∑ᵢ Σi⊗Vi), then 𝔼[R'R] = ∑ᵢ tr(Vi) × Σi. 

    # Let S = (R'R) / n
    mul!(model.storage_d_d_1, transpose(model.R), model.R, T(inv(n)), zero(T))

    trV = zeros(T, m)
    for j in 1:m
        trV[j] = tr(model.V[j])
    end

    for k in 1:m
        model.VarComp[k].Σ .= model.storage_d_d_1 .* (n / (trV[k] * m))
        if model.Σ_rank[k] < d
            rk = model.Σ_rank[k]
            eigtemp = eigen(model.VarComp[k].Σ; sortby = x -> -x)
            for j in 1:rk
                eigtemp.values[j] = max(1e-8, sqrt(eigtemp.values[j]))
                for i in 1:d
                    model.VarComp[k].F[i, j] = eigtemp.vectors[i, j] * eigtemp.values[j]
                end
            end
            BLAS.syrk!('L', 'N', -one(T), model.VarComp[k].F, one(T), model.VarComp[k].Σ)
            # trace fill
            fill!(parent(model.VarComp[k].Ψ), tr(model.VarComp[k].Σ) / d)
            # Residual fill
            # for j in 1:d
            #     model.VarComp[k].Ψ[j, j] = model.VarComp[k].Σ[j, j]
            # end
            # Update Σ with new estimate
            copyto!(model.VarComp[k].Σ, model.VarComp[k].Ψ)
            BLAS.syrk!('L', 'N', one(T), model.VarComp[k].F, one(T), model.VarComp[k].Σ)
            copytri!(model.VarComp[k].Σ, 'L')
        end
    end
    update_Ω!(model)
end

"""
    update_Σ!(model::MultiResponseVarianceComponentModel; algo::Symbol)

Update the variance component parameters `model.VarComp`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_Σ!(
    model :: MultiResponseVarianceComponentModel{T};
    algo  :: Symbol = :MM
    ) where T <: BlasReal
    d = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # update Ω⁻¹R, assuming Ω⁻¹ = model.storage_nd_nd
    copyto!(model.storage_nd_1, model.R)
    mul!(model.storage_nd_2, Ω⁻¹, model.storage_nd_1)
    copyto!(model.Ω⁻¹R, model.storage_nd_2)
    for k in 1:length(model.V)
        if model.Σ_rank[k] ≥ d
            update_Σk!(model, k, Val(algo))
        else
            # TODO: EM algorithm for structured variance components
            update_Σk!(model, k, model.Σ_rank[k], Val(algo))
        end
    end
    update_Ω!(model)
    # model.Σ
end

"""
    update_Σk!(model::MultiResponseVarianceComponentModel, k, Val(:MM))

MM update the `model.VarComp[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MultiResponseVarianceComponentModel{T},
    k     :: Integer,
          :: Val{:MM}
    ) where T <: BlasReal
    Σk  = model.VarComp[k].Σ
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k])) = the Mnj matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # lower Cholesky factor L of gradient
    _, info = LAPACK.potrf!('L', model.storage_d_d_1)
    info > 0 && throw("gradient of Σ[$k] is singular")
    # storage_d_d_2 = L' * Σ[k] * (R' * V[k] * R) * Σ[k] * L
    mul!(model.storage_n_d_1, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d_1)
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), model.storage_d_d_1, Σk)
    mul!(model.storage_d_d_3, model.storage_d_d_2, Σk)
    mul!(model.storage_d_d_2, transpose(Σk), model.storage_d_d_3)
    # Σ[k] = sqrtm(storage_d_d_2) for now
    # TODO: write custom function for allocation-free call to LAPACK.syev!
    vals, vecs = LAPACK.syev!('V', 'L', model.storage_d_d_2)
    @inbounds for j in eachindex(vals)
        if vals[j] > 0
            v = sqrt(sqrt(vals[j]))
            for i in axes(vecs, 1)
                vecs[i, j] *= v
            end
        else
            for i in axes(vecs, 1)
                vecs[i, j] = zero(T)
            end
        end
    end
    mul!(Σk, vecs, transpose(vecs))
    # inverse of Cholesky factor of gradient
    LAPACK.trtri!('L', 'N', model.storage_d_d_1)
    # update variance component Σ[k]
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), model.storage_d_d_1, Σk)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), model.storage_d_d_1, Σk)
    return Σk
end

"""
    update_Σk!(model::MultiResponseVarianceComponentModel, k, Val(:EM))

EM update the `model.VarComp[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MultiResponseVarianceComponentModel{T},
    k     :: Integer,
          :: Val{:EM}
    ) where T <: BlasReal
    Σk  = model.VarComp[k].Σ
    d   = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k])) = the Mnj matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # storage_d_d_2 = R' * V[k] * R
    mul!(model.storage_n_d_1, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d_1)
    # storage_d_d_2 = (R' * V[k] * R - Mk) / rk
    model.storage_d_d_2 .= (model.storage_d_d_2 .- model.storage_d_d_1) ./ model.V_rank[k]
    mul!(model.storage_d_d_1, model.storage_d_d_2, Σk)
    @inbounds for j in 1:d
        model.storage_d_d_1[j, j] += one(T)
    end
    mul!(Σk, copyto!(model.storage_d_d_2, Σk), model.storage_d_d_1)
    # enforce symmetry
    copytri!(Σk, 'U')
    return Σk
end

"""
    update_Σk!(model, k, rk, Val(:MM))

Update the parameters `model.F[k]` and `model.Ψ[:,k]` assuming a diagonal plus
low rank structure for `model.VarComp[k]`. Assumes covariance matrix `model.Ω` is 
available at `model.storage_nd_nd` and `model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MultiResponseVarianceComponentModel{T},
    k     :: Integer,
    rk    :: Integer,
          :: Val{:MM}
    ) where T <: BlasReal
    Σk  = model.VarComp[k].Σ
    Fk  = model.VarComp[k].F
    ψk  = parent(model.VarComp[k].Ψ)

    d   = size(Fk, 1)
    Ω⁻¹ = model.storage_nd_nd
    # M = storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    M = model.storage_d_d_1

    # N = storage_d_d_2 = R' * V[k] * R
    mul!(model.storage_n_d_1, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d_1)
    N = model.storage_d_d_2

    # Update Ψ[k]
    @inbounds for i in 1:d
        ψk[i] *= sqrt(N[i, i] / M[i, i])
    end
    # store √ψ for later
    model.storage_d .= sqrt.(ψk)

    # Store FₖFₖᵀ + Ψₖ₊₁ in Σₖ
    BLAS.syrk!('L', 'N', one(T), Fk, zero(T), Σk)
    copytri!(Σk, 'L')
    @inbounds for i in 1:d
        Σk[i, i] += ψk[i]
    end

    # C = Σ[k] * N * Σ[k]
    BLAS.symm!('L', 'L', one(T), Σk, N, zero(T), model.storage_d_d_4)
    BLAS.symm!('R', 'L', one(T), Σk, model.storage_d_d_4, zero(T), model.storage_d_d_3)
    C = model.storage_d_d_3

    # Update Hₖ = Ψ^(-1/2) Fₖ
    H = view(model.storage_d_d_4, :, 1:rk)
    copyto!(H, Fk)
    @inbounds for j in 1:rk
        for i in 1:d
            H[i, j] /= model.storage_d[i]
        end
    end

    # Evaluate M̃ = √Ψ * M * √Ψ and C̃ = (√Ψ)⁻¹ * M * (√Ψ)⁻¹ in-place
    @inbounds for j in 1:d
        for i in 1:d
            temp = model.storage_d[j] * model.storage_d[i]
            M[i, j] *= temp
            C[i, j] /= temp
        end
    end

    W = view(model.storage_d_d_5, 1:rk, 1:rk)
    if rk == 1
        W[1] = sum(abs2, H) + one(T)
    else
        BLAS.syrk!('L', 'T', one(T), H, zero(T), W)
        @inbounds for j in 1:rk
            W[j, j] += one(T)
        end
        LAPACK.potrf!('L', W)
    end

    H_Winv = view(model.storage_d_d_6, :, 1:rk)
    copyto!(H_Winv, H)
    if rk == 1
        H_Winv ./= W[1]
    else
        BLAS.trsm!('R', 'L', 'T', 'N', one(T), W, H_Winv)
        BLAS.trsm!('R', 'L', 'N', 'N', one(T), W, H_Winv)
    end

    # RHS = C̃ * (H / W)
    RHS = view(model.storage_d_d_7, :, 1:rk)
    BLAS.symm!('L', 'L', one(T), C, H_Winv, zero(T), RHS)

    A = M

    # B = W \ ((transpose(H) * C̃ * H) / W)
    B = view(model.storage_p_p, 1:rk, 1:rk)
    mul!(B, transpose(H_Winv), RHS)

    # Solving Sylvester Equation AX + XB = C
    λA, _ = LAPACK.syev!('V', 'L', A)
    λB, _ = LAPACK.syev!('V', 'L', B)

    # Dont need H_Winv anymore so its free storage
    RHS_temp = view(model.storage_d_d_6, :, 1:rk)
    mul!(RHS_temp, transpose(A), RHS)
    mul!(RHS, RHS_temp, B)

    @inbounds for j in 1:rk
        for i in 1:d
            RHS[i,j] /= (λA[i] + λB[j])
        end
    end

    mul!(RHS_temp, A, RHS)
    mul!(RHS, RHS_temp, transpose(B))

    # Update F[k]
    copyto!(Fk, RHS)
    @inbounds for j in 1:rk
        for i in 1:d
            Fk[i, j] *= model.storage_d[i]
        end
    end

    # update Σ[k]
    BLAS.syrk!('L', 'N', one(T), Fk, zero(T), Σk)
    @inbounds for i in 1:d
        Σk[i, i] += ψk[i]
    end
    copytri!(Σk, 'L')
    return Σk
end

"""
    update_B!(model::MultiResponseVarianceComponentModel)

Update the regression coefficients `model.B`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_B!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    G   = model.storage_pd_pd
    # Gram matrix G = (Id⊗X')Ω⁻¹(Id⊗X) = (X'Ω⁻¹ᵢⱼX)ᵢⱼ, 1 ≤ i,j ≤ d
    n, d, p = size(model.Y, 1), size(model.Y, 2), size(model.X, 2)
    # TODO: this block-wise accessing of Ω⁻¹ is pretty slow
    for j in 1:d
        Ωcidx = ((j - 1) * n + 1):(j * n)
        Gcidx = ((j - 1) * p + 1):(j * p)
        for i in 1:j
            Ωridx = ((i - 1) * n + 1):(i * n)
            Gridx = ((i - 1) * p + 1):(i * p)
            Ω⁻¹ᵢⱼ = view(Ω⁻¹, Ωridx, Ωcidx)
            Gᵢⱼ   = view(G  , Gridx, Gcidx)
            mul!(model.storage_n_p, Ω⁻¹ᵢⱼ, model.X)
            mul!(Gᵢⱼ, transpose(model.X), model.storage_n_p)
        end
    end
    copytri!(G, 'U')
    # (Id⊗X')Ω⁻¹vec(Y) = vec(X' * reshape(Ω⁻¹vec(Y), n, d))
    copyto!(model.storage_nd_1, model.Y)
    mul!(model.storage_nd_2, model.storage_nd_nd, model.storage_nd_1)
    copyto!(model.storage_n_d_1, model.storage_nd_2)
    mul!(model.storage_p_d, transpose(model.X), model.storage_n_d_1)
    # Cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Id⊗X')Ω⁻¹(Id⊗X) is singular")
    copyto!(model.storage_pd, model.storage_p_d)
    LAPACK.potrs!('U', G, model.storage_pd)
    copyto!(model.B, model.storage_pd)
    # update residuals R
    update_res!(model)
    model.B
end

"""
    loglikelihood!(model::MultiResponseVarianceComponentModel)

Overwrite `model.storage_nd_nd` by inverse of the covariance 
matrix `model.Ω`, overwrite `model.storage_nd` by `U' \\ vec(model.R)`, and 
return the log-likelihood. This function assumes `model.Ω` and `model.R` are 
already updated according to `model.VarComp` and `model.B`.
"""
function loglikelihood!(
    model::MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    copyto!(model.storage_nd_nd, model.Ω)
    # Cholesky of covariance Ω = U'U
    _, info = LAPACK.potrf!('U', model.storage_nd_nd)
    info > 0 && throw("covariance matrix Ω is singular")
    # storage_nd = U' \ vec(R)
    copyto!(model.storage_nd_1, model.R)
    BLAS.trsv!('U', 'T', 'N', model.storage_nd_nd, model.storage_nd_1)
    # assemble pieces for log-likelihood
    logl = sum(abs2, model.storage_nd_1) + length(model.storage_nd_1) * log(2π)
    @inbounds for i in 1:length(model.storage_nd_1)
        logl += 2log(model.storage_nd_nd[i, i])
    end
    # Ω⁻¹ from upper cholesky factor
    LAPACK.potri!('U', model.storage_nd_nd)
    copytri!(model.storage_nd_nd, 'U')
    logl /= -2
end

function update_res!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    # update R = Y - X B
    BLAS.gemm!('N', 'N', -one(T), model.X, model.B, one(T), copyto!(model.R, model.Y))
    model.R
end

function update_Ω!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    fill!(model.Ω, zero(T))
    @inbounds for k in 1:length(model.V)
        kron_axpy!(model.VarComp[k].Σ, model.V[k], model.Ω)
    end
    model.Ω
end

"""
    fisher_B!(model::MultiResponseVarianceComponentModel)

Compute the sampling variance-covariance of regression coefficients `model.B`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_B!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    G   = model.storage_pd_pd
    # Gram matrix G = (Id⊗X')Ω⁻¹(Id⊗X) = (X'Ω⁻¹ᵢⱼX)ᵢⱼ, 1 ≤ i,j ≤ d
    n, d, p = size(model.Y, 1), size(model.Y, 2), size(model.X, 2)
    for j in 1:d
        Ωcidx = ((j - 1) * n + 1):(j * n)
        Gcidx = ((j - 1) * p + 1):(j * p)
        for i in 1:j
            Ωridx = ((i - 1) * n + 1):(i * n)
            Gridx = ((i - 1) * p + 1):(i * p)
            Ω⁻¹ᵢⱼ = view(Ω⁻¹, Ωridx, Ωcidx)
            Gᵢⱼ   = view(G  , Gridx, Gcidx)
            mul!(model.storage_n_p, Ω⁻¹ᵢⱼ, model.X)
            mul!(Gᵢⱼ, transpose(model.X), model.storage_n_p)
        end
    end
    copytri!(G, 'U')
    copyto!(model.Bcov, pinv(G))
end

"""
    fisher_Σ!(model::MultiResponseVarianceComponentModel)

Compute the sampling variance-covariance of variance component estimates `model.VarComp`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_Σ!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    n, d, m = size(model.Y, 1), size(model.Y, 2), length(model.V)
    nd = n * d
    np = m * d^2
    for k in 1:m
        for j in 1:d
            mul!(view(model.storages_nd_nd[k], :, ((j - 1) * n + 1):(j * n)), 
                view(Ω⁻¹, :, ((j - 1) * n + 1):(j * n)), model.V[k])
        end
    end
    # E[-∂logl/∂vechΣⱼᵀ∂vechΣᵢ] = 1/2 Dd'⋅Wⱼ'(Ω⁻¹⊗Ω⁻¹)Wᵢ⋅Dd,
    # where Wᵢ = I_d⊗[(I_d⊗Vᵢ)Kdn]^(n) and Uᵢ = Wᵢ⋅Dd in manuscript
    Fisher = zeros(T, np, np)
    @inbounds for i in 1:np
        # compute 1/2 Wⱼ'(Ω⁻¹⊗Ω⁻¹)Wᵢ
        # keep track of indices for each column of Wᵢ
        k1     = div(i - 1, d^2) + 1 # 1 ≤ k1 ≤ m to choose Vₖ
        d2idx1 = mod1(i, d^2) # 1 ≤ d2idx ≤ d² to choose column of Wᵢ
        ddidx1 = div(d2idx1 - 1, d) # 0 ≤ ddidx ≤ d - 1 to choose columns of Ω⁻¹
        dridx1 = mod1(d2idx1, d) # 1 ≤ dridx ≤ d to choose columns of Ω⁻¹
        for j in i:np
            k2     = div(j - 1, d^2) + 1
            d2idx2 = mod1(j, d^2)
            ddidx2 = div(d2idx2 - 1, d)
            dridx2 = mod1(d2idx2, d)
            for (col, row) in enumerate((n * ddidx2 + 1):(n * ddidx2 + n))
                Fisher[i, j] += dot(view(model.storages_nd_nd[k1], row, (ddidx1 * n + 1):(ddidx1 * n + n)), 
                    view(model.storages_nd_nd[k2], (n * (dridx1 - 1) + 1):(n * dridx1), dridx2 * n - n + col))
            end
            Fisher[i, j] /= 2
        end
    end
    copytri!(Fisher, 'U')
    # compute 1/2 Dd'⋅Wⱼ'(Ω⁻¹⊗Ω⁻¹)Wᵢ⋅Dd
    vechF = zeros(T, (m * d * (d + 1)) >> 1, (m * d * (d + 1)) >> 1)
    D = duplication(d)
    for i in 1:m
        idx1 = Int(d * (d + 1) / 2 * (i - 1) + 1)
        idx2 = Int(d * (d + 1) / 2 * i)
        idx5, idx6 = d^2 * (i - 1) + 1, d^2 * i
        for j in i:m
            idx3 = Int(d * (d + 1) / 2 * (j - 1) + 1)
            idx4 = Int(d * (d + 1) / 2 * j)
            idx7, idx8 = d^2 * (j - 1) + 1, d^2 * j
            vechF[idx1:idx2, idx3:idx4] = D' * Fisher[idx5:idx6, idx7:idx8] * D
        end
    end
    copytri!(vechF, 'U')
    copyto!(model.Σcov, pinv(vechF))
end