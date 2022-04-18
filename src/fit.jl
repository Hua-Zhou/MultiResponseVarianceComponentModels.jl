"""
    fit!(model)

Fit a multivariate response variance component model by an MM algorithm.

# Positional arguments  
- `model`   : a `MultiResponseVarianceComponentModel` instance.  

# Keyword arguments
- `maxiter :: Integer`: maximum number of iterations. Default is `1000`.
- `reltol  :: Real`: relative tolerance for convergence. Default is `1e-6`.
- `verbose :: Bool`: display algorithmic information. Default is `false`.
- `init    :: Symbol`: initialization strategy. `:default` initialize by least squares. 
    `:user` uses user-supplied value at `model.Β` and `model.Σ`.
- `algo    :: Symbol`: optimization algorithm. `:MM` (default) or `EM`.
- `log     :: Bool`: record iterate history or not. Defaut is `false`.

# Output
- `model`   : result with `model.B` and `model.Σ` overwritten by the estimate.
- `logl`    : log-likelihood evaluated at final estimate.
- `history` : iterate history.
"""
function fit!(
    model   :: MultiResponseVarianceComponentModel{T};
    init    :: Symbol  = :default, # :default or :user
    algo    :: Symbol  = :MM, # :MM or :EM,
    maxiter :: Integer = 1000,
    reltol  :: Real    = 1e-6,
    verbose :: Bool    = false,
    log     :: Bool    = false
    ) where T <: BlasReal
    Y, X, V = model.Y, model.X, model.V
    # dimensions
    p, m = size(X, 2), length(V)
    # record iterate history if requested
    history          = ConvergenceHistory(partial = !log)
    history[:reltol] = reltol
    IterativeSolvers.reserve!(Int, history, :iter, maxiter)
    IterativeSolvers.reserve!(T, history, :logl, maxiter)
    IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)    
    # initialization
    tic = time()
    if init == :default
        if p > 0
            # estimate fixed effect coefficients Β by least squares (cholesky solve)
            copyto!(model.storage_p_p, model.xtx)
            _, info = LAPACK.potrf!('U', model.storage_p_p)
            info > 0 && throw("design matrix X is rank deficient")
            LAPACK.potrs!('U', model.storage_p_p, copyto!(model.Β, model.xty))
            # update residuls R
            update_res!(model)
        else
            # no fixed effects
            copy!(model.R, Y)
        end
        # initialization of variance components Σ[1], ..., Σ[m]
        # If R is MatrixNormal(0, Vi, Σi), i.e., vec(R) is Normal(0, Σi⊗Vi),
        # then E(R'R) = tr(Vi) * Σi. So we estimate Σi by R'R / tr(Vi)
        mul!(model.storage_d_d_1, transpose(model.R), model.R)
        for k in 1:m
            model.Σ[k] .= inv(tr(model.V[k])) .* model.storage_d_d_1
        end
        update_Ω!(model)
    elseif init == :user
        if p > 0; update_res!(model); else copy!(model.R, Y); end
        update_Ω!(model)
    else
        throw("unrecognize initialization method $init")
    end
    logl = loglikelihood!(model)
    toc = time()
    verbose && println("iter=0, logl=$logl")
    IterativeSolvers.nextiter!(history)
    push!(history, :iter    , 0)
    push!(history, :logl    , logl)
    push!(history, :itertime, toc - tic)
    # MM loop
    for iter in 1:maxiter
        IterativeSolvers.nextiter!(history)
        tic = time()
        # initial estiamte of Σ can be lousy, so we update Σ first in the 1st round
        p > 0 && iter > 1 && update_Β!(model)
        update_Σ!(model, algo = algo)
        logl_prev = logl
        logl = loglikelihood!(model)
        toc = time()
        verbose && println("iter=$iter, logl=$logl")
        push!(history, :iter    , iter)
        push!(history, :logl    , logl)
        push!(history, :itertime, toc - tic)
        if iter == maxiter
            @warn "maximum number of iterations $maxiter is reached."
            break
        end
        if abs(logl - logl_prev) < reltol * (abs(logl_prev) + 1)
            IterativeSolvers.setconv(history, true)
            break
        end
    end
    log && IterativeSolvers.shrink!(history)
    model, logl, history
end

"""
    update_Σ!(model)

Update the variance component parameters `model.Σ`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_Σ!(
    model :: MultiResponseVarianceComponentModel{T};
    algo  :: Symbol = :MM # :MM or :EM
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
            update_Σk!(model, k, model.Σ_rank[k])
        end
    end
    update_Ω!(model)
    model.Σ
end

"""
    update_Σk!(model, k, Val(:MM))

MM update of `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MultiResponseVarianceComponentModel{T},
    k     :: Integer,
          :: Val{:MM}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
    # = the Mnj matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # lower cholesky factor L of gradient
    _, info = LAPACK.potrf!('L', model.storage_d_d_1)
    info > 0 && throw("Gradient of Σ[$k] is singular")
    # storage_d_d_2 = L' * Σ[k] * (R' * V[k] * R) * Σ[k] * L
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    mul!(model.storage_d_d_3, model.storage_d_d_2, model.Σ[k])
    mul!(model.storage_d_d_2, transpose(model.Σ[k]), model.storage_d_d_3)
    # Σ[k] = sqrtm(storage_d_d_2) for now
    vals, vecs = eigen!(Symmetric(model.storage_d_d_2))
    @inbounds for j in 1:length(vals)
        if vals[j] > 0
            v = sqrt(sqrt(vals[j]))
            for i in 1:size(vecs, 1)
                vecs[i, j] *= v
            end
        else
            for i in 1:size(vecs, 1)
                vecs[i, j] = 0
            end
        end
    end
    mul!(model.Σ[k], vecs, transpose(vecs))
    # inverse of Cholesky factor of gradient
    LAPACK.trtri!('L', 'N', model.storage_d_d_1)
    # update variance component Σ[k]
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    model.Σ[k]
end

"""
    update_Σk!(model, k, Val(:EM))

EM update of `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MultiResponseVarianceComponentModel{T},
    k     :: Integer,
          :: Val{:EM}
    ) where T <: BlasReal
    d   = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
    # = the Mnj matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # storage_d_d_2 = R' * V[k] * R
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    # storage_d_d_2 = (R' * V[k] * R - Mk) / rk
    model.storage_d_d_2 .= 
        (model.storage_d_d_2 .- model.storage_d_d_1) ./ model.V_rank[k]
    mul!(model.storage_d_d_1, model.storage_d_d_2, model.Σ[k])
    @inbounds for j in 1:d
        model.storage_d_d_1[j, j] += 1
    end
    mul!(model.Σ[k], copyto!(model.storage_d_d_2, model.Σ[k]), model.storage_d_d_1)
    # enforce symmetry
    copytri!(model.Σ[k], 'U')
    model.Σ[k]
end

"""
    update_Β!(model)

Update the regression coefficients `model.Β`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_Β!(
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
    # (Id⊗X')Ω⁻¹vec(Y) = vec(X' * reshape(Ω⁻¹vec(Y), n, d))
    copyto!(model.storage_nd_1, model.Y)
    mul!(model.storage_nd_2, model.storage_nd_nd, model.storage_nd_1)
    copyto!(model.storage_n_d, model.storage_nd_2)
    mul!(model.storage_p_d, transpose(model.X), model.storage_n_d)
    # cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Id⊗X')Ω⁻¹(Id⊗X) is singular")
    copyto!(model.storage_pd, model.storage_p_d)
    LAPACK.potrs!('U', G, model.storage_pd)
    copyto!(model.Β, model.storage_pd)
    # update residuls R
    update_res!(model)
    model.Β
end

"""
    loglikelihood!(model::MultiResponseVarianceComponentModel)

Overwrite `model.storage_nd_nd` by inverse of the covariance 
matrix `model.Ω`, overwrite `model.storage_nd` by `U' \\ vec(model.R)`, and 
return the log-likelihood. This function assumes `model.Ω` and `model.R` are 
already updated according to `model.Σ` and `model.Β`.
"""
function loglikelihood!(
    model::MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    copyto!(model.storage_nd_nd, model.Ω)
    # cholesky of covariance Ω = U'U
    _, info = LAPACK.potrf!('U', model.storage_nd_nd)
    info > 0 && throw("covariance matrix Ω is singular")
    # storage_nd = U' \ vec(R)
    copyto!(model.storage_nd_1, model.R)
    BLAS.trsv!('U', 'T', 'N', model.storage_nd_nd, model.storage_nd_1)
    # assemble pieces for log-likelihood
    logl = norm(model.storage_nd_1)^2 + length(model.storage_nd_1) * log(2π)
    @inbounds for i in 1:length(model.storage_nd_1)
        logl += 2log(model.storage_nd_nd[i, i])
    end
    # Ω⁻¹ from upper cholesky factor
    LAPACK.potri!('U', model.storage_nd_nd)
    copytri!(model.storage_nd_nd, 'U')
    # return
    logl /= -2
end

function update_res!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    # update R = Y - X Β
    BLAS.gemm!('N', 'N', -one(T), model.X, model.Β, one(T), copyto!(model.R, model.Y))
    model.R
end

function update_Ω!(
    model :: MultiResponseVarianceComponentModel
    )
    fill!(model.Ω, 0)
    for k in 1:length(model.V)
        kron_axpy!(model.Σ[k], model.V[k], model.Ω)
    end
    model.Ω
end

