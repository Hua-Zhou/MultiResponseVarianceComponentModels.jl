"""
    fit!(model::SimpleMRVCModel)
    fit!(model::MRTVCModel)

Fit a multivariate response variance components model by MM or EM algorithm.

# Keyword arguments
```
maxiter::Int        maximum number of iterations; default 1000
reltol::Real        relative tolerance for convergence; default 1e-6
verbose::Bool       display algorithmic information; default true
init::Symbol        initialization strategy; :default initializes by least squares, while
    :user uses user-supplied values at model.B and model.Σ
algo::Symbol        optimization algorithm; :MM (default) or :EM (for SimpleMRVCModel)
log::Bool           record iterate history or not; default false
```

# Extended help
MM algorithm is provably faster than EM algorithm in this setting, so recommend trying 
MM algorithm first, which is by default, and switching to EM algorithm if there are 
convergence issues.
"""
function fit!(
    model   :: SimpleMRVCModel{T};
    maxiter :: Int     = 1000,
    reltol  :: T       = 1e-6,
    verbose :: Bool    = true,
    init    :: Symbol  = :default, # :default or :user
    algo    :: Symbol  = :MM,
    log     :: Bool    = false,
    ) where {T}
    Y, X = model.Y, model.X
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(model.VarComp)
    @info "Running $algo algorithm for ML estimation"
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
        S = model.storage_d_d_1
        mul!(S, transpose(model.R), model.R, inv(n), zero(T))
        for j in 1:m
            Vj = model.VarComp[j].V
            model.VarComp[j].Σ .= model.storage_d_d_1 * (n / (m * tr(Vj)))
            initialize!(model.VarComp[j])
        end
    elseif init == :user
        if p > 0
            update_res!(model)
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
    push!(history, :iter    , 0)
    push!(history, :logl    , logl)
    push!(history, :itertime, toc - tic)
    # MM loop
    for iter in 1:maxiter
        IterativeSolvers.nextiter!(history)
        tic = time()
        # initial estiamte of Σ[i] can be lousy, so we update Σ[i] first in the 1st round
        if p > 0 && iter > 1
            update_B!(model)
        end
        update_Σ!(model, algo = algo)
        logl_prev = logl
        logl = loglikelihood!(model)
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
            @info "Relative increase in objective function is less than $reltol\n"
            copyto!(model.logl, logl)
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
    #     # need Ω⁻¹ for B
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
    update_Σ!(model::SimpleMRVCModel)

Update the variance component parameters `model.Σ`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`. If
missing response, assume conditional variance `model.storage_n_miss_n_miss_1`
is precomputed.
"""
function update_Σ!(
    model    :: SimpleMRVCModel{T};
    algo     :: Symbol = :MM
    ) where T <: BlasReal
    d = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # update Ω⁻¹R, assuming Ω⁻¹ = model.storage_nd_nd
    mul!(model.storage_nd_2, Ω⁻¹, copyto!(model.storage_nd_1, model.R))
    copyto!(model.Ω⁻¹R, model.storage_nd_2)
    # if ymissing
    #     # compute Ω⁻¹P'CPΩ⁻¹
    #     nd = size(Ω⁻¹, 1)
    #     n_obs = nd - model.n_miss
    #     C = model.storage_n_miss_n_miss_1
    #     PΩ⁻¹ = model.storage_nd_nd_miss
    #     PΩ⁻¹ .= @view Ω⁻¹[model.P, :]
    #     copy!(model.storage_n_miss_n_obs_2,
    #         view(PΩ⁻¹, (n_obs + 1):nd, 1:n_obs))
    #     copy!(model.storage_n_miss_n_miss_2, 
    #         view(PΩ⁻¹, (n_obs + 1):nd, (n_obs + 1):nd))
    #     Ω⁻¹PtCPΩ⁻¹ = model.storage_nd_nd_miss
    #     mul!(model.storage_n_miss_n_obs_3, C, model.storage_n_miss_n_obs_2)
    #     mul!(view(Ω⁻¹PtCPΩ⁻¹, 1:n_obs, 1:n_obs), 
    #         transpose(model.storage_n_miss_n_obs_2), model.storage_n_miss_n_obs_3)
    #     mul!(view(Ω⁻¹PtCPΩ⁻¹, (n_obs + 1):nd, 1:n_obs), 
    #         transpose(model.storage_n_miss_n_miss_2), model.storage_n_miss_n_obs_3)
    #     mul!(model.storage_n_miss_n_miss_3, C, model.storage_n_miss_n_miss_2)
    #     mul!(view(Ω⁻¹PtCPΩ⁻¹, (n_obs + 1):nd, (n_obs + 1):nd), 
    #         transpose(model.storage_n_miss_n_miss_2), model.storage_n_miss_n_miss_3)
    #     copytri!(Ω⁻¹PtCPΩ⁻¹, 'L')    
    # end
    for k in eachindex(model.VarComp)
        update_M!(model.VarComp[k], Ω⁻¹)
        update_N!(model.VarComp[k], model.Ω⁻¹R)
        update_Σ!(model.VarComp[k], Val(algo))
    end
    update_Ω!(model)
    # model.Σ
end

"""
    update_Σk!(model::SimpleMRVCModel, k, Val(:EM))

EM update the `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` is precomputed.
"""
function update_Σk!(
    model :: SimpleMRVCModel{T},
    k     :: Integer,
          :: Val{:EM}
    ) where {T}
    d   = size(model.Y, 2)
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr[Ω⁻¹(Σ[k]⊗V[k])] = the M[k] matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1; sym = true)
    # storage_d_d_2 = R'V[k]R
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    # storage_d_d_2 = (R'V[k]R - M[k]) / rank[k]
    model.storage_d_d_2 .= (model.storage_d_d_2 .- model.storage_d_d_1) ./ model.V_rank[k]
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
    update_B!(model::SimpleMRVCModel)

Update the regression coefficients `model.B`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function update_B!(
    model :: SimpleMRVCModel{T}
    ) where {T}
    Ω⁻¹ = model.storage_nd_nd
    G   = model.storage_pd_pd
    # Gram matrix G = (Id ⊗ X)' Ω⁻¹(Id ⊗ X) = (X' Ω⁻¹ᵢⱼ X)ᵢⱼ, 1 ≤ i,j ≤ d
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
    mul!(model.storage_nd_2, Ω⁻¹, copyto!(model.storage_nd_1, model.Y))
    copyto!(model.storage_n_d, model.storage_nd_2)
    mul!(model.storage_p_d, transpose(model.X), model.storage_n_d)
    # Cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Id ⊗ X)ᵀ Ω⁻¹ (Id ⊗ X) is singular")
    copyto!(model.storage_pd, model.storage_p_d)
    LAPACK.potrs!('U', G, model.storage_pd)
    copyto!(model.B, model.storage_pd)
    # update residuals R
    update_res!(model)
    model.B
end

"""
    loglikelihood!(model::SimpleMRVCModel)

Overwrite `model.storage_nd_nd` by inverse of the covariance 
matrix `model.Ω`, overwrite `model.storage_nd` by `U' \\ vec(model.R)`, and 
return the log-likelihood. Assume `model.Ω` and `model.R` are 
already updated according to `model.Σ` and `model.B`.
"""
function loglikelihood!(
    model :: SimpleMRVCModel{T}
    ) where {T}
    copyto!(model.storage_nd_nd, model.Ω)
    # Cholesky of covariance Ω = U'U
    _, info = LAPACK.potrf!('U', model.storage_nd_nd)
    info > 0 && throw(PosDefException(info))
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

function update_res!(
    model :: SimpleMRVCModel{T}
    ) where {T}
    # update R = Y - XB
    BLAS.gemm!('N', 'N', -one(T), model.X, model.B, one(T), copyto!(model.R, model.Y))
    return model.R
end

function update_Ω!(model::SimpleMRVCModel{T}) where {T}
    fill!(model.Ω, zero(T))
    _update_Ω!(model.Ω, model.VarComp)
    return model.Ω
end

@generated function _update_Ω!(Ω::Matrix{T}, VarComp::VC) where {T, VC<:Tuple}
    K = length(VC.parameters)
    updates = [:(kron_axpy!(VarComp[$i].Σ, VarComp[$i].V, Ω)) for i in 1:K]
    quote
        $(Expr(:meta, :propagate_inbounds))
        begin
            $(updates...)
        end
    end
end

"""
    fisher_B!(model::SimpleMRVCModel)

Compute the sampling variance-covariance `model.Bcov` of regression coefficients `model.B`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_B!(
    model :: SimpleMRVCModel{T}
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
    fisher_Σ!(model::SimpleMRVCModel)

Compute the sampling variance-covariance `model.Σcov` of variance component estimates `model.Σ`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_Σ!(
    model :: SimpleMRVCModel{T}
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
    vechFisher = zeros(T, m * ◺(d), m * ◺(d))
    D = duplication(d)
    for i in 1:m
        idx1 = ◺(d) * (i - 1) + 1
        idx2 = ◺(d) * i
        idx5, idx6 = d^2 * (i - 1) + 1, d^2 * i
        for j in i:m
            idx3 = ◺(d) * (j - 1) + 1
            idx4 = ◺(d) * j
            idx7, idx8 = d^2 * (j - 1) + 1, d^2 * j
            vechFisher[idx1:idx2, idx3:idx4] = D' * Fisher[idx5:idx6, idx7:idx8] * D
        end
    end
    copytri!(vechFisher, 'U')
    copyto!(model.Σcov, pinv(vechFisher))
end