"""
    fit!(model::MRVCModel)

Fit a multivariate response variance components model by MM or EM algorithm.

# Keyword arguments
```
maxiter::Int        maximum number of iterations; default 1000
reltol::Real        relative tolerance for convergence; default 1e-8
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
    reltol::T     = 1e-8,
    verbose::Bool = true,
    init::Symbol  = :default, # :default or :user
    algo::Symbol  = :MM,
    log::Bool     = false,
) where {T}
    K = length(model.Ω.VarComp)
    Y, X = model.Y, model.X
    # dimensions
    n, d, p = size(Y, 1), size(Y, 2), size(X, 2)
    if model.reml == true
        project_null!(model)
        p = 0
    end

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
        # Fixed Effects
        if p > 0
            # initial fixed effect coefficients B by ordinary least squares (Cholesky solve)
            copyto!(model.storage_p_p, model.xtx)
            _, info = LAPACK.potrf!('U', model.storage_p_p)
            info > 0 && throw(PosDefException("XᵀX is singular."))
            LAPACK.potrs!('U', model.storage_p_p, copyto!(model.B, model.xty))
            # update residuals R
            update_residuals!(model)
        else
            # no fixed effects
            copyto!(model.R, Y)
        end
        # Variance Components Σ[1], …, Σ[K]
        M = Matrix{T}(undef, K, K)
        vech_RtVR = Matrix{T}(undef, ◺(d), K)
        VR = model.storage_n_d
        RtVR = model.storage_d_d_1
        for j in 1:K
            BLAS.symm!(
                'L', 'L', one(T), model.Ω.VarComp[j].V, model.R, zero(T), VR
            )
            mul!(RtVR, transpose(model.R), VR)
            vech!(view(vech_RtVR, :, j), RtVR)
            for i in j:K
                M[i, j] = dot(model.Ω.VarComp[i].V, model.Ω.VarComp[j].V)
            end
        end
        LAPACK.potrf!('L', M)
        BLAS.trsm!('R', 'L', 'T', 'N', one(T), M, vech_RtVR)
        BLAS.trsm!('R', 'L', 'N', 'N', one(T), M, vech_RtVR)
        for j in 1:K
            unvech!(model.storage_d_d_1, view(vech_RtVR, :, j))
            copytri!(model.storage_d_d_1, 'L')
            λ, Q = eigen(Symmetric(model.storage_d_d_1, :L))
            λ[λ .< reltol] .= reltol
            model.Ω.VarComp[j].Σ .= Q * Diagonal(λ) * transpose(Q)
        end
        initialize!(model.Ω.VarComp)
    elseif init == :user
        if p > 0
            update_residuals!(model)
        else
            copy!(model.R, Y)
        end
    else
        throw("Cannot recognize initialization method $init")
    end
    set_evaluated!(model.Ω, false)
    set_factorized!(model.Ω, false)
    # Calculate Ω from initial values of B and Σ[1], ..., Σ[K]
    update_Ω!(model)
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
        # initial estimate of Σ[i] can be lousy, 
        # so we update Σ[i] first in the 1st round
        if p > 0 && iter > 1
            update_B!(model)
        end
        update_residuals!(model)
        update_Σ!(model; algo=algo)
        set_evaluated!(model.Ω, false)
        set_factorized!(model.Ω, false)
        # update Ω and log-likelihood after updating Σ
        update_Ω!(model)
        logl_prev = model.logl[]
        logl = loglikelihood!(model; calc_Ωinv=true)
        logl_prev > logl &&
            @error "Log-likelihood decreased from $logl_prev to $logl"
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
            model.logl[] = logl
            finalize!(model.Ω)
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

"""
    loglikelihood!(model::MRVCModel)

Compute log-likelihood and store Ωinv in `model.storage_nd_nd`.
Assume `model.R` is already updated according to `model.Σ` and `model.B`.
"""
function loglikelihood!(model::MRVCModel{T}; calc_Ωinv::Bool=true) where {T}
    nd = length(model.storage_nd_1)
    # factorize Ω in-place to obtain Cholesky factor
    factorize!(model.Ω)
    # storage_nd_1 = L \ vec(R)
    copyto!(model.storage_nd_1, model.R)
    BLAS.trsv!('L', 'N', 'N', model.Ω.factor, model.storage_nd_1)
    # assemble pieces for log-likelihood
    loglik = nd * log2π + logdet(model.Ω) + sum(abs2, model.storage_nd_1)
    loglik /= -2
    if calc_Ωinv == true
        # Ωinv from lower Cholesky factor
        copyto!(model.storage_nd_nd, model.Ω.factor)
        LAPACK.potri!('L', model.storage_nd_nd)
        copytri!(model.storage_nd_nd, 'L')
    end
    return model.logl[] = loglik
end

function update_residuals!(model::MRVCModel{T}) where {T}
    if model.reml == true
        return nothing
    else
        # update R = Y - XB
        mul!(copyto!(model.R, model.Y), model.X, model.B, -one(T), one(T))
        return model.R
    end
end

"""
    update_B!(model::MRVCModel)

Update the regression coefficients `model.B`, assuming inverse of
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_B!(model::MRVCModel{T}) where {T}
    if model.reml == true
        return nothing
    else
        Ωinv = model.storage_nd_nd
        G    = model.storage_pd_pd
        # Gram matrix G = (Id ⊗ X)ᵀ Ωinv (Id ⊗ X) = (Xᵀ Ωinvᵢⱼ X)ᵢⱼ, 1 ≤ i,j ≤ d
        n, d, p = size(model.Y, 1), size(model.Y, 2), size(model.X, 2)
        for j in 1:d
            Ωcidx = ((j - 1) * n + 1):(j * n)
            Gcidx = ((j - 1) * p + 1):(j * p)
            for i in 1:j
                Ωridx   = ((i - 1) * n + 1):(i * n)
                Gridx   = ((i - 1) * p + 1):(i * p)
                Ωinv_ij = view(Ωinv, Ωridx, Ωcidx)
                Gij     = view(G, Gridx, Gcidx)
                mul!(model.storage_n_p, Ωinv_ij, model.X)
                mul!(Gij, transpose(model.X), model.storage_n_p)
            end
        end
        copytri!(G, 'U')
        # (Id ⊗ X') Ω \ vec(Y) = vec(X' * reshape(Ω \ vec(Y), n, d))
        BLAS.symv!(
            'L',
            one(T),
            Ωinv,
            copyto!(model.storage_nd_1, model.Y),
            zero(T),
            model.storage_nd_2,
        )
        copyto!(model.storage_n_d, model.storage_nd_2)
        mul!(model.storage_p_d, transpose(model.X), model.storage_n_d)
        # Cholesky solve
        _, info = LAPACK.potrf!('U', G)
        info > 0 && throw("Gram matrix (Id ⊗ X)ᵀ Ωinv (Id ⊗ X) is singular")
        copyto!(model.storage_pd, model.storage_p_d)
        LAPACK.potrs!('U', G, model.storage_pd)
        return copyto!(model.B, model.storage_pd)
    end
end

"""
    update_Ω!(model::MRVCModel)

Evaluate the covariance matrix `model.Ω` based on the current variance component
estimates in model.Ω.VarComp.Σ
"""
function update_Ω!(model::MRVCModel{T}) where {T}
    materialize!(model.Ω)
    set_evaluated!(model.Ω, true)
    set_factorized!(model.Ω, false)
    return model.Ω
end

"""
    update_Σ!(model::MRVCModel)

Update the variance component parameters `model.Σ`, assuming inverse of
covariance matrix `model.Ω` is available at `model.storage_nd_nd`. If
missing response, assume conditional variance `model.storage_n_miss_n_miss_1`
is precomputed.
"""
function update_Σ!(model::MRVCModel{T}; algo::Symbol=:MM) where {T<:BlasReal}
    d = size(model.Y, 2)
    Ωinv = model.storage_nd_nd
    ΩL = model.Ω.factor
    # update Ωinv_R, assuming Ω.factor is the cholesky factor of Ω
    if is_factorized(model.Ω) == false
        throw("Ω needs to be factorized to update Σ")
    end
    # Ωinv_R is needed to update N = R̃ᵀVR̃, where R̃ = vec⁻¹(Ωinv vec(R))
    BLAS.symv!(
        'L',
        one(T),
        Ωinv,
        copyto!(model.storage_nd_1, model.R),
        zero(T),
        model.storage_nd_2,
    )
    copyto!(model.Ωinv_R, model.storage_nd_2)
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
    # Generated functions using meta-programming
    # type of each variance component is known at compile time
    update_M!(model.Ω.VarComp, Ωinv)
    update_N!(model.Ω.VarComp, model.Ωinv_R)
    update_Σ!(model.Ω.VarComp, Val(algo))
    return nothing
end
