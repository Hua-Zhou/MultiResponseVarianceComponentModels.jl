"""
    fit!(model::MRVCModel)

Fit a multivariate response variance components model by MM or EM algorithm.

# Keyword arguments
```
maxiter::Int        maximum number of iterations; default 1000
reltol::Real        relative tolerance for convergence; default 1e-6
verbose::Bool       display algorithmic information; default true
init::Symbol        initialization strategy; :default initializes by least squares, while
    :user uses user supplied values at model.B and model.Σ
algo::Symbol        optimization algorithm; :MM (default) or EM
log::Bool           record iterate history or not; default false
```
"""
function fit!(
    model   :: MRVCModel{T};
    maxiter :: Integer = 1000,
    reltol  :: Real = 1e-6,
    verbose :: Bool = true,
    init    :: Symbol = :default, # :default or :user
    algo    :: Symbol  = :MM,
    log     :: Bool = false,
    ) where T <: BlasReal
    if model.ymissing
        @assert algo == :MM "only MM algorithm is possible for missing response"
    end
    Y, X, V = model.Y, model.X, model.V
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(V)
    if model.reml
        @info "Running $(algo) algorithm for REML estimation"
    else
        @info "Running $(algo) algorithm for ML estimation"
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
        if p > 0
            # estimate fixed effect coefficients B by ordinary least squares (Cholesky solve)
            copyto!(model.storage_p_p, model.xtx)
            _, info = LAPACK.potrf!('U', model.storage_p_p)
            info > 0 && throw("Design matrix X is rank deficient")
            LAPACK.potrs!('U', model.storage_p_p, copyto!(model.B, model.xty))
            # update residuals R
            update_res!(model)
        else
            # no fixed effects
            copy!(model.R, Y)
        end
        # initialization of variance components Σ[1], ..., Σ[m]
        # if R is MatrixNormal(0, V[i], Σ[i]), i.e., vec(R) is Normal(0, Σ[i]⊗V[i]),
        # then E(R'R) = tr(V[i]) * Σ[i], so we estimate Σ[i] by R'R / tr(V[i])
        mul!(model.storage_d_d_1, transpose(model.R), model.R)
        for k in 1:m
            model.Σ[k] .= inv(tr(model.V[k])) .* model.storage_d_d_1
        end
        update_Ω!(model)
    elseif init == :user
        if p > 0; update_res!(model); else copy!(model.R, Y); end
        update_Ω!(model)
    else
        throw("Cannot recognize initialization method $init")
    end
    if model.ymissing
        # update conditional variance
        C = model.storage_n_miss_n_miss_1
        PΩPt = model.storage_nd_nd_miss
        PΩPt .= @view model.Ω[model.P, model.P]
        nd = size(model.Ω, 1)
        n_obs = nd - model.n_miss    
        sweep!(PΩPt, 1:n_obs)
        copytri!(PΩPt, 'U')
        copy!(model.storage_n_miss_n_obs_1, 
            view(PΩPt, (n_obs + 1):nd, 1:n_obs)) # for conditional mean
        copy!(model.storage_n_miss_n_miss_1, 
            view(PΩPt, (n_obs + 1):nd, (n_obs + 1):nd)) # conditional variance
        logl = NaN
    else
        logl = loglikelihood!(model)
    end
    toc = time()
    if !model.ymissing
        verbose && println("iter = 0, logl = $logl")
        IterativeSolvers.nextiter!(history)
        push!(history, :iter    , 0)
        push!(history, :logl    , logl)
        push!(history, :itertime, toc - tic)
    end
    # MM loop
    for iter in 1:maxiter
        IterativeSolvers.nextiter!(history)
        tic = time()
        # initial estiamte of Σ[i] can be lousy, so we update Σ[i] first in the 1st round
        if p > 0 && iter > 1 && model.ymissing
            update_B_miss!(model)
        elseif p > 0 && iter > 1
            update_B!(model)
        end
        update_Σ!(model, algo = algo, ymissing = model.ymissing)
        logl_prev = logl
        model.ymissing ? logl = loglikelihood_miss!(model) : logl = loglikelihood!(model)
        toc = time()
        verbose && println("iter = $iter, logl = $logl")
        push!(history, :iter    , iter)
        push!(history, :logl    , logl)
        push!(history, :itertime, toc - tic)
        if iter == maxiter
            @warn "Maximum number of iterations $maxiter is reached"
            break
        end
        if abs(logl - logl_prev) < reltol * (abs(logl_prev) + 1)
            @info "Updates converged!"
            copyto!(model.logl, logl)
            IterativeSolvers.setconv(history, true)
            if model.se
                @info "Calculating standard errors"
                fisher_B!(model)
                fisher_Σ!(model)
            end
            break
        end
    end
    if model.reml
        update_Ω_reml!(model)
        # need Ω⁻¹ for B
        copyto!(model.storage_nd_nd_reml, model.Ω_reml)
        # Cholesky of covariance Ω = U'U
        _, info = LAPACK.potrf!('U', model.storage_nd_nd_reml)
        info > 0 && throw("Covariance matrix Ω is singular")
        LAPACK.potri!('U', model.storage_nd_nd_reml)
        copytri!(model.storage_nd_nd_reml, 'U')
        update_B_reml!(model)
        copyto!(model.logl_reml, loglikelihood_reml!(model))
        model.se ? fisher_B_reml!(model) : nothing
    end
    log && IterativeSolvers.shrink!(history)
    history
end

"""
    update_Σ!(model::MRVCModel)

Update the variance component parameters `model.Σ`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`. If
missing response, assume conditional variance `model.storage_n_miss_n_miss_1`
is precomputed.
"""
function update_Σ!(
    model    :: MRVCModel{T};
    algo     :: Symbol = :MM,
    ymissing :: Bool = false
    ) where T <: BlasReal
    d = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # update Ω⁻¹R, assuming Ω⁻¹ = model.storage_nd_nd
    copyto!(model.storage_nd_1, model.R)
    mul!(model.storage_nd_2, Ω⁻¹, model.storage_nd_1)
    copyto!(model.Ω⁻¹R, model.storage_nd_2)
    if ymissing
        # compute Ω⁻¹P'CPΩ⁻¹
        nd = size(Ω⁻¹, 1)
        n_obs = nd - model.n_miss
        C = model.storage_n_miss_n_miss_1
        PΩ⁻¹ = model.storage_nd_nd_miss
        PΩ⁻¹ .= @view Ω⁻¹[model.P, :]
        copy!(model.storage_n_miss_n_obs_2,
            view(PΩ⁻¹, (n_obs + 1):nd, 1:n_obs))
        copy!(model.storage_n_miss_n_miss_2, 
            view(PΩ⁻¹, (n_obs + 1):nd, (n_obs + 1):nd))
        Ω⁻¹PtCPΩ⁻¹ = model.storage_nd_nd_miss
        mul!(model.storage_n_miss_n_obs_3, C, model.storage_n_miss_n_obs_2)
        mul!(view(Ω⁻¹PtCPΩ⁻¹, 1:n_obs, 1:n_obs), 
            transpose(model.storage_n_miss_n_obs_2), model.storage_n_miss_n_obs_3)
        mul!(view(Ω⁻¹PtCPΩ⁻¹, (n_obs + 1):nd, 1:n_obs), 
            transpose(model.storage_n_miss_n_miss_2), model.storage_n_miss_n_obs_3)
        mul!(model.storage_n_miss_n_miss_3, C, model.storage_n_miss_n_miss_2)
        mul!(view(Ω⁻¹PtCPΩ⁻¹, (n_obs + 1):nd, (n_obs + 1):nd), 
            transpose(model.storage_n_miss_n_miss_2), model.storage_n_miss_n_miss_3)
        copytri!(Ω⁻¹PtCPΩ⁻¹, 'L')    
    end
    for k in 1:length(model.V)
        if model.Σ_rank[k] ≥ d
            if ymissing
                update_Σk_miss!(model, k, Val(algo))
            else
                update_Σk!(model, k, Val(algo))
            end
        else
            update_Σk!(model, k, model.Σ_rank[k])
        end
    end
    update_Ω!(model)
    model.Σ
end

"""
    update_Σk!(model::MRVCModel, k, Val(:MM))

MM update the `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` is precomputed.
"""
function update_Σk!(
    model :: MRVCModel{T},
    k     :: Integer,
          :: Val{:MM}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr[Ω⁻¹(Σ[k]⊗V[k])] = the M[k] matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # lower Cholesky factor L of gradient M[i]
    _, info = LAPACK.potrf!('L', model.storage_d_d_1)
    info > 0 && throw("Gradient of Σ[$k] is singular")
    # storage_d_d_2 = L'Σ[k](R'V[k]R)Σ[k]L
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
    # inverse of Cholesky factor of gradient M[k]
    LAPACK.trtri!('L', 'N', model.storage_d_d_1)
    # update variance component Σ[k]
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    model.Σ[k]
end

"""
    update_Σk_miss!(model::MRVCModel, k, Val(:MM))

MM update the `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, 
`model.Ω⁻¹R` is precomputed, and Ω⁻¹P'CPΩ⁻¹ is available at
`model.storage_nd_nd_miss`.
"""
function update_Σk_miss!(
    model :: MRVCModel{T},
    k     :: Integer,
          :: Val{:MM}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    Ω⁻¹PtCPΩ⁻¹ = model.storage_nd_nd_miss
    # storage_d_d_miss = gradient of tr[Ω⁻¹P'CPΩ⁻¹(Σ[k]⊗V[k])] = the M*[k] matrix in manuscript
    kron_reduction!(Ω⁻¹PtCPΩ⁻¹, model.V[k], model.storage_d_d_miss, true)
    # storage_d_d_1 = gradient of tr[Ω⁻¹(Σ[k]⊗V[k])] = the M[k] matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # lower Cholesky factor L of M[k]
    _, info = LAPACK.potrf!('L', model.storage_d_d_1)
    info > 0 && throw("Gradient of Σ[$k] is singular")
    # storage_d_d_2 = L'Σ[k](R'V[k]R + M*[k])Σ[k]L
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    model.storage_d_d_2 .= model.storage_d_d_2 .+ model.storage_d_d_miss
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
    # inverse of Cholesky factor of gradient M[k]
    LAPACK.trtri!('L', 'N', model.storage_d_d_1)
    # update variance component Σ[k]
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), model.storage_d_d_1, model.Σ[k])
    model.Σ[k]
end

"""
    update_Σk!(model::MRVCModel, k, Val(:EM))

EM update the `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` is precomputed.
"""
function update_Σk!(
    model :: MRVCModel{T},
    k     :: Integer,
          :: Val{:EM}
    ) where T <: BlasReal
    d   = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr[Ω⁻¹(Σ[k]⊗V[k])] = the M[k] matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # storage_d_d_2 = R'V[k]R
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    # storage_d_d_2 = (R'V[k]R - M[k]) / rank[k]
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
    update_B!(model::MRVCModel)

Update the regression coefficients `model.B`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_B!(
    model :: MRVCModel{T}
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
    update_B_miss!(model::MRVCModel)

Update the regression coefficients `model.B`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd` and 
`model.storage_n_miss_n_obs_1` for conditional mean is precomputed.
"""
function update_B_miss!(
    model :: MRVCModel{T}
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
    # compute imputed response
    nd = size(Ω⁻¹, 1)
    n_obs = nd - model.n_miss
    mul!(model.R, model.X, model.B)
    copyto!(model.storage_nd_1, model.R)
    model.storage_nd_2 .= @view model.storage_nd_1[model.P] # expected mean
    copy!(model.storage_n_obs, view(model.storage_nd_2, 1:n_obs))
    copy!(model.storage_n_miss, view(model.storage_nd_2, (n_obs + 1):nd))
    model.storage_n_obs .= model.Y_obs - model.storage_n_obs
    BLAS.gemv!('N', one(T), model.storage_n_miss_n_obs_1, model.storage_n_obs, one(T), model.storage_n_miss) # conditional mean
    copyto!(model.storage_nd_2, model.Y_obs)
    copy!(view(model.storage_nd_2, (n_obs + 1):nd), model.storage_n_miss)
    model.storage_nd_1 .= @view model.storage_nd_2[model.invP]
    copyto!(model.Y, model.storage_nd_1)
    # (Id⊗X')Ω⁻¹vec(Y) = vec(X' * reshape(Ω⁻¹vec(Y), n, d))
    mul!(model.storage_nd_2, model.storage_nd_nd, model.storage_nd_1)
    copyto!(model.storage_n_d, model.storage_nd_2)
    mul!(model.storage_p_d, transpose(model.X), model.storage_n_d)
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

function update_B_reml!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd_reml
    G   = model.storage_pd_pd_reml
    # Gram matrix G = (Id⊗X')Ω⁻¹(Id⊗X) = (X'Ω⁻¹ᵢⱼX)ᵢⱼ, 1 ≤ i,j ≤ d
    n, d, p = size(model.Y_reml, 1), size(model.Y_reml, 2), size(model.X_reml, 2)
    for j in 1:d
        Ωcidx = ((j - 1) * n + 1):(j * n)
        Gcidx = ((j - 1) * p + 1):(j * p)
        for i in 1:j
            Ωridx = ((i - 1) * n + 1):(i * n)
            Gridx = ((i - 1) * p + 1):(i * p)
            Ω⁻¹ᵢⱼ = view(Ω⁻¹, Ωridx, Ωcidx)
            Gᵢⱼ   = view(G  , Gridx, Gcidx)
            mul!(model.storage_n_p_reml, Ω⁻¹ᵢⱼ, model.X_reml)
            mul!(Gᵢⱼ, transpose(model.X_reml), model.storage_n_p_reml)
        end
    end
    copytri!(G, 'U')
    # (Id⊗X')Ω⁻¹vec(Y) = vec(X' * reshape(Ω⁻¹vec(Y), n, d))
    copyto!(model.storage_nd_1_reml, model.Y_reml)
    mul!(model.storage_nd_2_reml, model.storage_nd_nd_reml, model.storage_nd_1_reml)
    copyto!(model.storage_n_d_reml, model.storage_nd_2_reml)
    mul!(model.storage_p_d_reml, transpose(model.X_reml), model.storage_n_d_reml)
    # Cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Id⊗X')Ω⁻¹(Id⊗X) is singular")
    copyto!(model.storage_pd_reml, model.storage_p_d_reml)
    LAPACK.potrs!('U', G, model.storage_pd_reml)
    copyto!(model.B_reml, model.storage_pd_reml)
    # update residuals R
    update_res_reml!(model)
    model.B_reml
end

"""
    loglikelihood!(model::MRVCModel)

Overwrite `model.storage_nd_nd` by inverse of the covariance 
matrix `model.Ω`, overwrite `model.storage_nd` by `U' \\ vec(model.R)`, and 
return the log-likelihood. Assume `model.Ω` and `model.R` are 
already updated according to `model.Σ` and `model.B`.
"""
function loglikelihood!(
    model::MRVCModel{T}
    ) where T <: BlasReal
    copyto!(model.storage_nd_nd, model.Ω)
    # Cholesky of covariance Ω = U'U
    _, info = LAPACK.potrf!('U', model.storage_nd_nd)
    info > 0 && throw("Covariance matrix Ω is singular")
    # storage_nd = U' \ vec(R)
    copyto!(model.storage_nd_1, model.R)
    BLAS.trsv!('U', 'T', 'N', model.storage_nd_nd, model.storage_nd_1)
    # assemble pieces for log-likelihood
    logl = sum(abs2, model.storage_nd_1) + length(model.storage_nd_1) * log(2π)
    @inbounds for i in 1:length(model.storage_nd_1)
        logl += 2log(model.storage_nd_nd[i, i])
    end
    # Ω⁻¹ from upper Cholesky factor
    LAPACK.potri!('U', model.storage_nd_nd)
    copytri!(model.storage_nd_nd, 'U')
    logl /= -2
end

"""
    loglikelihood_miss!(model::MRVCModel)

Overwrite `model.storage_nd_nd` by inverse of the covariance matrix `model.Ω`, 
overwrite `model.storage_nd` by `U' \\ vec(model.R)`, overwrite
`model.storage_n_miss_n_miss_1` by conditional variance, precompute 
`model.storage_n_miss_n_obs_1` for conditional mean, and return the value of
surrogate Q-function of log-likelihood. Assume `model.Ω` and `model.R` are already 
updated according to `model.Σ` and `model.B`.
"""
function loglikelihood_miss!(
    model::MRVCModel{T}
    ) where T <: BlasReal
    copyto!(model.storage_nd_nd, model.Ω)
    # Cholesky of covariance Ω = U'U
    _, info = LAPACK.potrf!('U', model.storage_nd_nd)
    info > 0 && throw("Covariance matrix Ω is singular")
    # storage_nd = U' \ vec(R)
    copyto!(model.storage_nd_1, model.R)
    BLAS.trsv!('U', 'T', 'N', model.storage_nd_nd, model.storage_nd_1)
    # assemble pieces for log-likelihood
    logl = sum(abs2, model.storage_nd_1) + length(model.storage_nd_1) * log(2π)
    @inbounds for i in 1:length(model.storage_nd_1)
        logl += 2log(model.storage_nd_nd[i, i])
    end
    # Ω⁻¹ from upper Cholesky factor
    LAPACK.potri!('U', model.storage_nd_nd)
    copytri!(model.storage_nd_nd, 'U')
    # compute tr(PΩ⁻¹PtC) for surrogate function of log-likelihood
    C = model.storage_n_miss_n_miss_1
    PΩ⁻¹Pt = model.storage_nd_nd_miss
    Ω⁻¹ = model.storage_nd_nd
    PΩ⁻¹Pt .= @view Ω⁻¹[model.P, model.P]
    nd = size(Ω⁻¹, 1)
    n_obs = nd - model.n_miss
    logl += dot(view(PΩ⁻¹Pt, (n_obs + 1):nd, (n_obs + 1):nd), C)
    # update conditional variance
    PΩPt = model.storage_nd_nd_miss
    PΩPt .= @view model.Ω[model.P, model.P]
    sweep!(PΩPt, 1:n_obs)
    copytri!(PΩPt, 'U')
    copy!(model.storage_n_miss_n_obs_1, 
        view(PΩPt, (n_obs + 1):nd, 1:n_obs)) # for conditional mean
    copy!(model.storage_n_miss_n_miss_1, 
        view(PΩPt, (n_obs + 1):nd, (n_obs + 1):nd)) # conditional variance
    logl /= -2
end

function loglikelihood_reml!(
    model::MRVCModel{T}
    ) where T <: BlasReal
    copyto!(model.storage_nd_nd_reml, model.Ω_reml)
    # Cholesky of covariance Ω = U'U
    _, info = LAPACK.potrf!('U', model.storage_nd_nd_reml)
    info > 0 && throw("Covariance matrix Ω is singular")
    # storage_nd = U' \ vec(R)
    copyto!(model.storage_nd_1_reml, model.R_reml)
    BLAS.trsv!('U', 'T', 'N', model.storage_nd_nd_reml, model.storage_nd_1_reml)
    # assemble pieces for log-likelihood
    logl = sum(abs2, model.storage_nd_1_reml) + length(model.storage_nd_1_reml) * log(2π)
    @inbounds for i in 1:length(model.storage_nd_1_reml)
        logl += 2log(model.storage_nd_nd_reml[i, i])
    end
    # Ω⁻¹ from upper Cholesky factor
    LAPACK.potri!('U', model.storage_nd_nd_reml)
    copytri!(model.storage_nd_nd_reml, 'U')
    logl /= -2
end

function update_res!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    # update R = Y - XB
    BLAS.gemm!('N', 'N', -one(T), model.X, model.B, one(T), copyto!(model.R, model.Y))
    model.R
end

function update_res_reml!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    # update R = Y - XB
    BLAS.gemm!('N', 'N', -one(T), model.X_reml, model.B_reml, one(T), copyto!(model.R_reml, model.Y_reml))
    model.R
end

function update_Ω!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    fill!(model.Ω, zero(T))
    @inbounds for k in 1:length(model.V)
        kron_axpy!(model.Σ[k], model.V[k], model.Ω)
    end
    model.Ω
end

function update_Ω_reml!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    fill!(model.Ω_reml, zero(T))
    @inbounds for k in 1:length(model.V_reml)
        kron_axpy!(model.Σ[k], model.V_reml[k], model.Ω_reml)
    end
    model.Ω_reml
end

"""
    fisher_B!(model::MRVCModel)

Compute the sampling variance-covariance `model.Bcov` of regression coefficients `model.B`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_B!(
    model :: MRVCModel{T}
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

function fisher_B_reml!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd_reml
    G   = model.storage_pd_pd_reml
    # Gram matrix G = (Id⊗X')Ω⁻¹(Id⊗X) = (X'Ω⁻¹ᵢⱼX)ᵢⱼ, 1 ≤ i,j ≤ d
    n, d, p = size(model.Y_reml, 1), size(model.Y_reml, 2), size(model.X_reml, 2)
    for j in 1:d
        Ωcidx = ((j - 1) * n + 1):(j * n)
        Gcidx = ((j - 1) * p + 1):(j * p)
        for i in 1:j
            Ωridx = ((i - 1) * n + 1):(i * n)
            Gridx = ((i - 1) * p + 1):(i * p)
            Ω⁻¹ᵢⱼ = view(Ω⁻¹, Ωridx, Ωcidx)
            Gᵢⱼ   = view(G  , Gridx, Gcidx)
            mul!(model.storage_n_p_reml, Ω⁻¹ᵢⱼ, model.X_reml)
            mul!(Gᵢⱼ, transpose(model.X_reml), model.storage_n_p_reml)
        end
    end
    copytri!(G, 'U')
    copyto!(model.Bcov_reml, pinv(G))
end

"""
    fisher_Σ!(model::MRVCModel)

Compute the sampling variance-covariance of variance component estimates `model.Σ`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_Σ!(
    model :: MRVCModel{T}
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
    # E[-∂logl/∂vechΣ[j]'∂vechΣ[i] = 1/2 Dd'W[j]'(Ω⁻¹⊗Ω⁻¹)W[i]Dd,
    # where W[i] = Id⊗[(Id⊗V[i])Kdn]^(n) and U[i] = W[i]⋅Dd in manuscript
    Fisher = zeros(T, np, np)
    @inbounds for i in 1:np
        # compute 1/2 W[j]'(Ω⁻¹⊗Ω⁻¹)W[i]
        # keep track of indices for each column of W[i]
        k1     = div(i - 1, d^2) + 1 # 1 ≤ k1 ≤ m to choose V[k]
        d2idx1 = mod1(i, d^2) # 1 ≤ d2idx ≤ d² to choose column of W[i]
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
    # compute 1/2 Dd'W[j]'(Ω⁻¹⊗Ω⁻¹)W[i]Dd
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