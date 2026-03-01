"""
    fit!(model::MRVCModel)

Fit a multivariate response variance components model by MM or EM algorithm.

# Keyword arguments
```
maxiter::Int        maximum number of iterations; default 1000
reltol::Real        relative tolerance for convergence; default 1e-6
verbose::Bool       display algorithmic information; default true
init::Symbol        initialization strategy; :default initializes by least squares, while
    :user uses user-supplied values at model.B and model.Σ
algo::Symbol        optimization algorithm; :MM (default) or :EM
log::Bool           record iterate history or not; default false
```

# Extended help
MM algorithm is provably faster than EM algorithm in this setting, so recommend trying
MM algorithm first, which is by default, and switching to EM algorithm if there are
convergence issues.
"""
function fit!(
    model::MRVCModel{T};
    maxiter::Int  = 1000,
    reltol::T     = 1e-6,
    verbose::Bool = true,
    init::Symbol  = :default, # :default or :user
    algo::Symbol  = :MM,
    log::Bool     = false,
) where {T}
    Y, X = model.Y, model.X
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(model.VarComp)
    @info "Running $algo algorithm for ML estimation"
    # record iterate history if requested
    history          = ConvergenceHistory(; partial=!log)
    history[:reltol] = reltol
    IterativeSolvers.reserve!(Int, history, :iter, maxiter + 1)
    IterativeSolvers.reserve!(T, history, :logl, maxiter + 1)
    IterativeSolvers.reserve!(Float64, history, :itertime, maxiter + 1)
    # initialization
    tic = time()
    if init == :default
        if p > 0
            # initial fixed effect coefficients B by ordinary least squares (Cholesky solve)
            copyto!(model.storage_p_p, model.xtx)
            _, info = LAPACK.potrf!('U', model.storage_p_p)
            info > 0 && throw("Design matrix X is rank deficient")
            LAPACK.potrs!('U', model.storage_p_p, copyto!(model.B, model.xty))
            # update residuals R
            update_residuals!(model)
        else
            # no fixed effects
            copy!(model.R, Y)
        end
        # initialization of variance components Σ[1], ..., Σ[m]
        # if R is MatrixNormal(0, V[i], Σ[i]), i.e., vec(R) is Normal(0, Σ[i]⊗V[i]),
        # then E(R'R) = tr(V[i]) * Σ[i], so we estimate Σ[i] by R'R / tr(V[i])
        RtR = model.storage_d_d_1
        mul!(RtR, transpose(model.R), model.R, inv(n), zero(T))
        for j in 1:m
            Vj = model.VarComp[j].V
            model.VarComp[j].Σ .= RtR * (n / (m * tr(Vj)))
            initialize!(model.VarComp[j])
        end
    elseif init == :user
        if p > 0
            update_residuals!(model)
        else
            copy!(model.R, Y)
        end
    else
        throw("Cannot recognize initialization method $init")
    end
    update_Ω!(model)
    # if model.ymissing
    #     # update conditional variance
    #     C = model.storage_n_miss_n_miss_1
    #     PΩPt = model.storage_nd_nd_miss
    #     PΩPt .= @view model.Ω[model.P, model.P]
    #     nd = size(model.Ω, 1)
    #     n_obs = nd - model.n_miss    
    #     sweep!(PΩPt, 1:n_obs)
    #     copytri!(PΩPt, 'U')
    #     copy!(model.storage_n_miss_n_obs_1, 
    #         view(PΩPt, (n_obs + 1):nd, 1:n_obs)) # for conditional mean
    #     copy!(model.storage_n_miss_n_miss_1, 
    #         view(PΩPt, (n_obs + 1):nd, (n_obs + 1):nd)) # conditional variance
    #     logl = NaN
    # else
    # end
    logl = loglikelihood!(model)
    toc = time()
    verbose && println("iter = 0, logl = $logl")
    IterativeSolvers.nextiter!(history)
    push!(history, :iter, 0)
    push!(history, :logl, logl)
    push!(history, :itertime, toc - tic)
    # MM loop
    for iter in 1:maxiter
        IterativeSolvers.nextiter!(history)
        tic = time()
        # initial estiamte of Σ[i] can be lousy, so we update Σ[i] first in the 1st round
        if p > 0 && iter > 1
            update_B!(model)
        end
        update_Σ!(model; algo=algo)
        logl_prev = logl
        logl = loglikelihood!(model)
        toc = time()
        verbose && println("iter = $iter, logl = $logl")
        push!(history, :iter, iter)
        push!(history, :logl, logl)
        push!(history, :itertime, toc - tic)
        if iter == maxiter
            @warn "Maximum number of iterations $maxiter is reached"
            break
        end
        if abs(logl - logl_prev) < reltol * (abs(logl_prev) + 1)
            @info "Relative change in objective function is less than $reltol\n"
            copyto!(model.logl, logl)
            finalize!(model.VarComp)
            IterativeSolvers.setconv(history, true)
            # if model.se
            #     @info "Calculating standard errors"
            #     fisher_B!(model)
            #     fisher_Σ!(model)
            # end
            break
        end
    end
    # if model.reml
    #     update_Ω_reml!(model)
    #     # need Ωinv for B
    #     copyto!(model.storage_nd_nd_reml, model.Ω_reml)
    #     # Cholesky of covariance Ω = U'U
    #     _, info = LAPACK.potrf!('U', model.storage_nd_nd_reml)
    #     info > 0 && throw("Covariance matrix Ω is singular")
    #     LAPACK.potri!('U', model.storage_nd_nd_reml)
    #     copytri!(model.storage_nd_nd_reml, 'U')
    #     update_B_reml!(model)
    #     copyto!(model.logl_reml, loglikelihood_reml!(model))
    #     model.se ? fisher_B_reml!(model) : nothing
    # end
    log && IterativeSolvers.shrink!(history)
    return history
end

@generated function finalize!(VarComps::VC) where {VC<:Tuple}
    K = length(VC.parameters)
    updates = [:(finalize!(VarComps[$i])) for i in 1:K]
    quote
        $(Expr(:meta, :propagate_inbounds))
        begin
            $(updates...)
        end
    end
end

"""
    loglikelihood!(model::MRVCModel)

Compute log-likelihood and store Ωinv in `model.storage_nd_nd`.
Assume `model.R` is already updated according to `model.Σ` and `model.B`.
"""
function loglikelihood!(model::MRVCModel{T}) where {T}
    ΩL = model.Ω.factor
    # materialize Ω and calculate its cholesky factor
    factorize!(model.Ω)
    # copy cholesky factor to storage for inversion
    copyto!(model.storage_nd_nd, ΩL)

    # storage_nd = L \ vec(R)
    copyto!(model.storage_nd_1, model.R)
    BLAS.trsv!('L', 'N', 'N', ΩL, model.storage_nd_1)

    # assemble pieces for log-likelihood
    logl = sum(abs2, model.storage_nd_1) + length(model.storage_nd_1) * log(2π)
    @inbounds for i in 1:length(model.storage_nd_1)
        logl += 2log(model.storage_nd_nd[i, i])
    end

    # Ωinv from lower Cholesky factor
    LAPACK.potri!('L', model.storage_nd_nd)
    copytri!(model.storage_nd_nd, 'L')
    return logl /= -2
end


"""
    update_B!(model::MRVCModel)

Update the regression coefficients `model.B`, assuming inverse of
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_B!(
    model::MRVCModel{T}
) where {T}
    Ωinv = model.storage_nd_nd
    G   = model.storage_pd_pd
    # Gram matrix G = (Id ⊗ X)' Ωinv(Id ⊗ X) = (X' Ωinvᵢⱼ X)ᵢⱼ, 1 ≤ i,j ≤ d
    n, d, p = size(model.Y, 1), size(model.Y, 2), size(model.X, 2)
    for j in 1:d
        Ωcidx = ((j - 1) * n + 1):(j * n)
        Gcidx = ((j - 1) * p + 1):(j * p)
        for i in 1:j
            Ωridx = ((i - 1) * n + 1):(i * n)
            Gridx = ((i - 1) * p + 1):(i * p)
            Ωinvᵢⱼ = view(Ωinv, Ωridx, Ωcidx)
            Gᵢⱼ   = view(G, Gridx, Gcidx)
            mul!(model.storage_n_p, Ωinvᵢⱼ, model.X)
            mul!(Gᵢⱼ, transpose(model.X), model.storage_n_p)
        end
    end
    copytri!(G, 'U')
    # (Id⊗X')Ωinvvec(Y) = vec(X' * reshape(Ωinvvec(Y), n, d))
    mul!(model.storage_nd_2, Ωinv, copyto!(model.storage_nd_1, model.Y))
    copyto!(model.storage_n_d, model.storage_nd_2)
    mul!(model.storage_p_d, transpose(model.X), model.storage_n_d)
    # Cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Id ⊗ X)ᵀ Ωinv (Id ⊗ X) is singular")
    copyto!(model.storage_pd, model.storage_p_d)
    LAPACK.potrs!('U', G, model.storage_pd)
    copyto!(model.B, model.storage_pd)
    # update residuals R
    update_residuals!(model)
    return model.B
end



function update_residuals!(model::MRVCModel{T}) where {T}
    # update R = Y - XB
    mul!(model.R, model.X, model.B, -one(T), one(T))
    return model.R
end

"""
    update_Ω!(model::MRVCModel)

Invalidate Ω factorization. Will be recomputed on next use.
"""
function update_Ω!(model::MRVCModel{T}) where {T}
    invalidate!(model.Ω)
    return model.Ω
end

"""
    update_Σ!(model::MRVCModel)

Update the variance component parameters `model.Σ`, assuming inverse of
covariance matrix `model.Ω` is available at `model.storage_nd_nd`. If
missing response, assume conditional variance `model.storage_n_miss_n_miss_1`
is precomputed.
"""
function update_Σ!(
    model::MRVCModel{T};
    algo::Symbol=:MM,
) where {T<:BlasReal}
    d = size(model.Y, 2)
    ΩL = model.Ω.factor
    Ωinv = model.storage_nd_nd
    # update Ωinv_R, assuming Ω.factor is the cholesky factor of Ω
    if is_factorized(model.Ω) == false
        throw("Ω needs to be factorized before updating Σ")
    end
    LAPACK.potrs!('L', ΩL, copyto!(model.Ωinv_R, model.R))
    # if ymissing
    #     # compute ΩinvP'CPΩinv
    #     nd = size(Ωinv, 1)
    #     n_obs = nd - model.n_miss
    #     C = model.storage_n_miss_n_miss_1
    #     PΩinv = model.storage_nd_nd_miss
    #     PΩinv .= @view Ωinv[model.P, :]
    #     copy!(model.storage_n_miss_n_obs_2,
    #         view(PΩinv, (n_obs + 1):nd, 1:n_obs))
    #     copy!(model.storage_n_miss_n_miss_2, 
    #         view(PΩinv, (n_obs + 1):nd, (n_obs + 1):nd))
    #     ΩinvPtCPΩinv = model.storage_nd_nd_miss
    #     mul!(model.storage_n_miss_n_obs_3, C, model.storage_n_miss_n_obs_2)
    #     mul!(view(ΩinvPtCPΩinv, 1:n_obs, 1:n_obs), 
    #         transpose(model.storage_n_miss_n_obs_2), model.storage_n_miss_n_obs_3)
    #     mul!(view(ΩinvPtCPΩinv, (n_obs + 1):nd, 1:n_obs), 
    #         transpose(model.storage_n_miss_n_miss_2), model.storage_n_miss_n_obs_3)
    #     mul!(model.storage_n_miss_n_miss_3, C, model.storage_n_miss_n_miss_2)
    #     mul!(view(ΩinvPtCPΩinv, (n_obs + 1):nd, (n_obs + 1):nd), 
    #         transpose(model.storage_n_miss_n_miss_2), model.storage_n_miss_n_miss_3)
    #     copytri!(ΩinvPtCPΩinv, 'L')    
    # end
    for k in eachindex(model.VarComp)
        update_M!(model.VarComp[k], Ωinv)
        update_N!(model.VarComp[k], model.Ωinv_R)
        update_Σ!(model.VarComp[k], Val(algo))
    end
    return update_Ω!(model)
    # model.Σ
end

