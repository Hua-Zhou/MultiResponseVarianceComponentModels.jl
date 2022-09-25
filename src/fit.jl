"""
    fit!(model::MultiResponseVarianceComponentModel)

Fit a multivariate response variance component model by an MM or EM algorithm.

# Positional arguments
- `model`: a `MultiResponseVarianceComponentModel` instance.  

# Keyword arguments
- `maxiter::Integer`: maximum number of iterations. Default is `1000`.
- `reltol::Real`: relative tolerance for convergence. Default is `1e-6`.
- `verbose::Bool`: display algorithmic information. Default is `false`.
- `init::Symbol`: initialization strategy. `:default` initialize by least squares.
    `:user` uses user supplied value at `model.B` and `model.Σ`.
- `algo::Symbol`: optimization algorithm. `:MM` (default) or `EM`.
- `log::Bool`: record iterate history or not. Defaut is `false`.
- `reml::Bool`: REML instead of ML estimation. Default is `false`.

# Output
- `history`: iterate history.
"""
function fit!(
    model   :: MultiResponseVarianceComponentModel{T};
    maxiter :: Integer = 1000,
    reltol  :: Real = 1e-6,
    verbose :: Bool = false,
    init    :: Symbol = :default, # :default or :user
    algo    :: Symbol  = :MM,
    log     :: Bool = false,
    reml    :: Bool = false
    ) where T <: BlasReal
    Y, X, V = model.Y, model.X, model.V
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(V)
    @info "n = $(n)"
    @info "d = $(d)"
    @info "p = $(p)"
    @info "m = $(m)"
    if reml
        Ỹ, Ṽ, _ = project_null(model.Y, model.X, model.V)
        modelf = MultiResponseVarianceComponentModel(Ỹ, Ṽ)
        @info("running $(algo) algorithm for REML estimation")
    else
        modelf = model
        @info("running $(algo) algorithm for ML estimation")
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
            copyto!(modelf.storage_p_p, modelf.xtx)
            _, info = LAPACK.potrf!('U', modelf.storage_p_p)
            info > 0 && throw("design matrix X is rank deficient")
            LAPACK.potrs!('U', modelf.storage_p_p, copyto!(modelf.B, modelf.xty))
            # update residuals R
            update_res!(modelf)
        else
            # no fixed effects
            copy!(modelf.R, Y)
        end
        # initialization of variance components Σ[1], ..., Σ[m]
        # If R is MatrixNormal(0, Vi, Σi), i.e., vec(R) is Normal(0, Σi⊗Vi),
        # then E(R'R) = tr(Vi) * Σi. So we estimate Σi by R'R / tr(Vi)
        mul!(modelf.storage_d_d_1, transpose(modelf.R), modelf.R)
        for k in 1:m
            modelf.Σ[k] .= inv(tr(modelf.V[k])) .* modelf.storage_d_d_1
        end
        update_Ω!(modelf)
    elseif init == :user
        if p > 0; update_res!(modelf); else copy!(modelf.R, Y); end
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
        # initial estiamte of Σ can be lousy, so we update Σ first in the 1st round
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
            fisher_Σ!(modelf)
            IterativeSolvers.setconv(history, true)
            break
        end
    end
    if reml
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
    end
    log && IterativeSolvers.shrink!(history)
    return history
end

"""
    update_Σ!(model::MultiResponseVarianceComponentModel)

Update the variance component parameters `model.Σ`, assuming inverse of 
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
            update_Σk!(model, k, model.Σ_rank[k])
        end
    end
    update_Ω!(model)
    model.Σ
end

"""
    update_Σk!(model::MultiResponseVarianceComponentModel, k, Val(:MM))

MM update the `model.Σ[k]` assuming it has full rank `d`, inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd`, and 
`model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MultiResponseVarianceComponentModel{T},
    k     :: Integer,
          :: Val{:MM}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    # storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k])) = the Mnj matrix in manuscript
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    # lower Cholesky factor L of gradient
    _, info = LAPACK.potrf!('L', model.storage_d_d_1)
    info > 0 && throw("gradient of Σ[$k] is singular")
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
    update_Σk!(model::MultiResponseVarianceComponentModel, k, Val(:EM))

EM update the `model.Σ[k]` assuming it has full rank `d`, inverse of 
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
    # storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k])) = the Mnj matrix in manuscript
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
    loglikelihood!(model::MultiResponseVarianceComponentModel)

Overwrite `model.storage_nd_nd` by inverse of the covariance 
matrix `model.Ω`, overwrite `model.storage_nd` by `U' \\ vec(model.R)`, and 
return the log-likelihood. This function assumes `model.Ω` and `model.R` are 
already updated according to `model.Σ` and `model.B`.
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
    logl = norm(model.storage_nd_1)^2 + length(model.storage_nd_1) * log(2π)
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
    model :: MultiResponseVarianceComponentModel
    )
    fill!(model.Ω, 0)
    for k in 1:length(model.V)
        kron_axpy!(model.Σ[k], model.V[k], model.Ω)
    end
    model.Ω
end

"""
    fisher_Σ!(model::MultiResponseVarianceComponentModel)

Compute the sampling variance-covariance of variance component estimates `model.Σ`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_Σ!(
    model :: MultiResponseVarianceComponentModel{T}
    ) where T <: BlasReal
    Ω⁻¹ = model.storage_nd_nd
    n, d, m = size(model.Y, 1), size(model.Y, 2), length(model.V)
    nd = n * d
    np = m * d^2
    # E[-∂logl/∂vechΣⱼᵀ∂vechΣᵢ] = 1/2 D_d'⋅Wⱼ(Ω⁻¹⊗Ω⁻¹)Wᵢ⋅D_d,
    # where Wᵢ = I_d⊗[(I_d⊗Vᵢ)K_dn]^(n) 
    Fisher = zeros(T, np, np)
    @inbounds for i in 1:np
        # keep track of indices for each column of Wᵢ, Wⱼ
        midx1 = div(i - 1, d^2) + 1 # 1 ≤ midx1 ≤ m to choose Vᵢ
        d2idx1 = mod1(i, d^2) # 1 ≤ d2idx1 ≤ d² to choose column of Wᵢ
        ddidx1 = div(d2idx1 - 1, d) # 0 ≤ ddidx1 ≤ d - 1 to choose columns of Ω⁻¹
        dridx1 = mod1(d2idx1, d) # 1 ≤ dridx1 ≤ d
        copyto!(model.storage_nd_n_1, view(Ω⁻¹, :, (ddidx1 * n + 1):(ddidx1 * n + n)))
        BLAS.gemm!('N', 'N', one(T), model.storage_nd_n_1, model.V[midx1], zero(T), model.storage_nd_n_2)
        lidx1 = n * (dridx1 - 1) + 1
        lidx2 = n * dridx1
        for j in i:np
            midx2 = div(j - 1, d^2) + 1
            d2idx2 = mod1(j, d^2)
            ddidx2 = div(d2idx2 - 1, d)
            dridx2 = mod1(d2idx2, d)
            copyto!(model.storage_nd_n_1, view(Ω⁻¹, :, (dridx2 * n - n + 1):(dridx2 * n)))
            BLAS.gemm!('N', 'N', one(T), model.storage_nd_n_1, model.V[midx2], zero(T), model.storage_nd_n_3)
            ridx1 = n * ddidx2 + 1
            ridx2 = n * ddidx2 + n
            for (col, row) in enumerate(ridx1:ridx2)
                Fisher[i, j] += dot(view(model.storage_nd_n_2, row, :), 
                    view(model.storage_nd_n_3, lidx1:lidx2, col))
            end
            Fisher[i, j] /= 2
        end
    end
    copytri!(Fisher, 'U')
    vechF = zeros((m * d * (d + 1)) >> 1, (m * d * (d + 1)) >> 1)
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
    LinearAlgebra.copytri!(vechF, 'U')
    copyto!(model.Σcov, pinv(vechF))
end

function calculate_p(
    model1::MultiResponseVarianceComponentModel,
    model0::MultiResponseVarianceComponentModel
    )
    df = length(model1.V) - length(model0.V)
    @assert df > 0
    lrt = 2 * (model1.logl[1] - model0.logl[1])
    coefs = [2.0^-df * binomial(df, i) for i in 1:df]
    ps = [ccdf(Chisq(i), lrt) for i in 1:df]
    sum(coefs .* ps)
end

function calculate_h2(model::MultiResponseVarianceComponentModel)
    m, d = length(model.Σ), size(model.Σ[1], 1)
    h² = zeros(m, d)
    h²se = zeros(m, d)
    Σ = zeros(size(model.Σ[1]))
    for i in 1:m
        Σ += model.Σ[i]
    end
    counter = 1
    l = binomial(d, 2) + d
    for j in 1:d
        ind = collect(counter:l:3l)
        Σcov = model.Σcov[ind, ind]
        Dh² = zeros(m, m)
        for i in 1:(m - 1)
            h²[i, j] = model.Σ[i][j, j] / Σ[j, j]
            for k in 1:m
                if k == i 
                    Dh²[i, k] = (Σ[j, j] - model.Σ[i][j, j]) / Σ[j, j]^2
                else
                    Dh²[i, k] = - model.Σ[i][j, j] / Σ[j, j]^2
                end
            end
        end
        h²[m, j] = (Σ[j, j] - model.Σ[m][j, j]) / Σ[j, j]
        for k in 1:m
            if k == m
                Dh²[m, k] = - (Σ[j, j] - model.Σ[m][j, j]) / Σ[j, j]^2
            else
                Dh²[m, k] = model.Σ[m][j, j] / Σ[j, j]^2
            end
        end
        h²se[:, j] = sqrt.(diag(Dh² * Σcov * Dh²'))
        counter += (d - j + 1)
    end
    d == 1 ? (vec(h²), vec(h²se)) : (h², h²se)
end

function calculate_rg(model::MultiResponseVarianceComponentModel)
    m, d = length(model.Σ), size(model.Σ[1], 1)
    @assert d > 1
    r₉ = [Matrix{Float64}(undef, d, d) for _ in 1:m]
    r₉se = [Matrix{Float64}(I, d, d) for _ in 1:m]
    for i in 1:m
        W = Diagonal(model.Σ[i])
        for j in 1:d
            W[j, j] = 1 / sqrt(W[j, j])
        end
        r₉[i] = W * model.Σ[i] * W
        ind = fill((binomial(d, 2) + d) * (i - 1) + 1, d)
        for j in 1:(d - 1)
            ind[j + 1] = ind[j] + (d - j + 1)
        end
        counter = (binomial(d, 2) + d) * (i - 1)
        for j in 1:d
            for k in j:d
                if k == j
                    counter += 1
                    continue
                else
                    Σcov = model.Σcov[[counter, ind[k], ind[j]], [counter, ind[k], ind[j]]]
                    ∇r₉ = [1 / sqrt(model.Σ[i][j, j] * model.Σ[i][k, k]),
                        -0.5 * model.Σ[i][k, j] / sqrt(model.Σ[i][j, j] * model.Σ[i][k, k]^3),
                        -0.5 * model.Σ[i][k, j] / sqrt(model.Σ[i][j, j]^3 * model.Σ[i][k, k])]
                    r₉se[i][k, j] = sqrt(∇r₉' * Σcov * ∇r₉)
                    counter += 1
                end
            end
        end
    end
    [copytri!(r₉se[i], 'L') for i in 1:m]
    r₉, r₉se
end