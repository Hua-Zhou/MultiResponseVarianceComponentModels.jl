function fit!(
    model   :: MRTVCModel{T};
    maxiter :: Integer = 1000,
    reltol  :: Real = 1e-6,
    verbose :: Bool = true,
    init    :: Symbol = :default,
    algo    :: Symbol = :MM,
    log     :: Bool = false,
    ) where T <: BlasReal
    Y, X, V = model.Y, model.X, model.V
    # dimensions
    n, d, p, m = size(Y, 1), size(Y, 2), size(X, 2), length(V)
    if model.reml
        @info "Running $algo algorithm for REML estimation"
    else
        @info "Running $algo algorithm for ML estimation"
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
            BLAS.gemm!('N', 'N', -one(T), model.X, model.B, one(T), copyto!(model.R̃Φ, model.Y))
            update_res!(model)
        else
            copy!(model.R̃Φ, model.Y)
            copy!(model.R̃, model.Ỹ)
        end
        # initialization of variance components Σ[1], Σ[2]
        # if R is MatrixNormal(0, V[i], Σ[i]), i.e., vec(R) is Normal(0, Σ[i]⊗V[i]),
        # then E(R'R) = tr(V[i]) * Σ[i], so we estimate Σ[i] by R'R / tr(V[i])
        mul!(model.storage_d_d_1, transpose(model.R̃Φ), model.R̃Φ)
        for k in 1:2
            model.Σ[k] .= inv(tr(model.V[k])) .* model.storage_d_d_1
        end
        update_Φ!(model)
        # update R̃Φ = (Ỹ - X̃B)Φ
        mul!(model.R̃Φ, model.R̃, model.Φ)
    elseif init == :user
        update_res!(model)
        update_Φ!(model)
        mul!(model.R̃Φ, model.R̃, model.Φ)
    else
        throw("Cannot recognize initialization method $init")
    end
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
            update_res!(model)
        end
        update_Σ!(model)
        update_Φ!(model)
        mul!(model.R̃Φ, model.R̃, model.Φ)
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
        update_B_reml!(model)
        update_res_reml!(model)
        mul!(model.R̃Φ_reml, model.R̃_reml, model.Φ)
        copyto!(model.logl_reml, loglikelihood_reml!(model))
        model.se ? fisher_B_reml!(model) : nothing
    end
    log && IterativeSolvers.shrink!(history)
    history
end

function update_Σ!(
    model :: MRTVCModel{T};
    ) where T <: BlasReal
    n, d = size(model.Ỹ, 1), size(model.Ỹ, 2)
    fill!(model.storage_d_1, zero(T))
    fill!(model.storage_d_2, zero(T))
    @inbounds for j in 1:d
        λj = model.Λ[j]
        for i in 1:n
            tmp = one(T) / (model.D[i] * λj + one(T))
            model.R̃Φ[i, j] *= tmp
            model.storage_d_2[j] += tmp
        end
        model.storage_d_2[j] = sqrt(model.storage_d_2[j])
    end
    mul!(model.N2tN2, transpose(model.R̃Φ), model.R̃Φ)
    @inbounds for j in 1:d
        λj = model.Λ[j]
        for i in 1:n
            tmp = model.D[i]
            model.R̃Φ[i, j] *= sqrt(tmp) * λj
            model.storage_d_1[j] += tmp / (tmp * λj + one(T))
        end
        model.storage_d_1[j] = sqrt(model.storage_d_1[j])
    end
    mul!(model.N1tN1, transpose(model.R̃Φ), model.R̃Φ)
    Φinv = inv(model.Φ)
    # update Σ[1]
    lmul!(Diagonal(model.storage_d_1), model.N1tN1)
    rmul!(model.N1tN1, Diagonal(model.storage_d_1))
    vals, vecs = eigen!(Symmetric(model.N1tN1))
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
    lmul!(Diagonal(one(T) ./ model.storage_d_1), vecs)
    mul!(model.storage_d_d_1, transpose(Φinv), vecs)
    mul!(model.Σ[1], model.storage_d_d_1, transpose(model.storage_d_d_1))
    # update Σ[2]
    lmul!(Diagonal(model.storage_d_2), model.N2tN2)
    rmul!(model.N2tN2, Diagonal(model.storage_d_2))
    vals, vecs = eigen!(Symmetric(model.N2tN2))
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
    lmul!(Diagonal(one(T) ./ model.storage_d_2), vecs)
    mul!(model.storage_d_d_1, transpose(Φinv), vecs)
    mul!(model.Σ[2], model.storage_d_d_1, transpose(model.storage_d_d_1))
    model.Σ
end

function update_Φ!(
    model :: MRTVCModel{T};
    ) where T <: BlasReal
    copy!(model.storage_d_d_1, model.Σ[1])
    copy!(model.storage_d_d_2, model.Σ[2])
    Λ, Φ = eigen!(Symmetric(model.storage_d_d_1), Symmetric(model.storage_d_d_2))
    copy!(model.Λ, Λ)
    copy!(model.Φ, Φ)
    copyto!(model.logdetΣ2, logdet(model.Σ[2]))
end

function update_res!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    # update R̃ = Ỹ - X̃B
    BLAS.gemm!('N', 'N', -one(T), model.X̃, model.B, one(T), copyto!(model.R̃, model.Ỹ))
    model.R̃
end

function update_res_reml!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    # update R̃ = Ỹ - X̃B
    BLAS.gemm!('N', 'N', -one(T), model.X̃_reml, model.B_reml, one(T), copyto!(model.R̃_reml, model.Ỹ_reml))
    model.R̃
end

function loglikelihood!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    n, d = size(model.Ỹ, 1), size(model.Ỹ, 2)
    # assemble pieces for log-likelihood
    logl = n * d * log(2π) + n * model.logdetΣ2[1] + d * model.logdetV2
    @inbounds for j in 1:d
        λj = model.Λ[j]
        @simd for i in 1:n
            tmp = model.D[i] * λj + one(T)
            logl += log(tmp) + inv(tmp) * model.R̃Φ[i, j]^2
        end
    end
    logl /= -2
end

function loglikelihood_reml!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    n, d = size(model.Ỹ_reml, 1), size(model.Ỹ_reml, 2)
    # assemble pieces for log-likelihood
    logl = n * d * log(2π) + n * model.logdetΣ2[1] + d * model.logdetV2_reml
    @inbounds for j in 1:d
        λj = model.Λ[j]
        @simd for i in 1:n
            tmp = model.D_reml[i] * λj + one(T)
            logl += log(tmp) + inv(tmp) * model.R̃Φ_reml[i, j]^2
        end
    end
    logl /= -2
end

function update_B!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    mul!(model.ỸΦ, model.Ỹ, model.Φ)
    # Gram matrix G = (Φ'⊗X̃)'(Λ⊗D + Ind)⁻¹(Φ'⊗X̃)
    G = model.storage_pd_pd
    fill!(model.storage_nd_pd, zero(T))
    kron_axpy!(transpose(model.Φ), model.X̃, model.storage_nd_pd)
    fill!(model.storage_nd_1, zero(T))
    kron_axpy!(model.Λ, model.D, model.storage_nd_1)
    @inbounds @simd for i in eachindex(model.storage_nd_1)
        model.storage_nd_1[i] = one(T) / sqrt(model.storage_nd_1[i] + one(T))
    end
    lmul!(Diagonal(model.storage_nd_1), model.storage_nd_pd)
    mul!(G, transpose(model.storage_nd_pd), model.storage_nd_pd)
    # (Φ'⊗X̃)'(Λ⊗D + Ind)⁻¹vec(ỸΦ)
    copyto!(model.storage_nd_2, model.ỸΦ)
    model.storage_nd_2 .= model.storage_nd_1 .* model.storage_nd_2
    mul!(model.storage_pd, transpose(model.storage_nd_pd), model.storage_nd_2)
    # Cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Φ'⊗X̃)'(Λ⊗D + Ind)⁻¹(Φ'⊗X̃) is singular")
    LAPACK.potrs!('U', G, model.storage_pd)
    copyto!(model.B, model.storage_pd)
    model.B
end

function update_B_reml!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    mul!(model.ỸΦ_reml, model.Ỹ_reml, model.Φ)
    # Gram matrix G = (Φ'⊗X̃)'(Λ⊗D + Ind)⁻¹(Φ'⊗X̃)
    G = model.storage_pd_pd_reml
    fill!(model.storage_nd_pd_reml, zero(T))
    kron_axpy!(transpose(model.Φ), model.X̃_reml, model.storage_nd_pd_reml)
    fill!(model.storage_nd_1_reml, zero(T))
    kron_axpy!(model.Λ, model.D_reml, model.storage_nd_1_reml)
    @inbounds @simd for i in eachindex(model.storage_nd_1_reml)
        model.storage_nd_1_reml[i] = one(T) / sqrt(model.storage_nd_1_reml[i] + one(T))
    end
    lmul!(Diagonal(model.storage_nd_1_reml), model.storage_nd_pd_reml)
    mul!(G, transpose(model.storage_nd_pd_reml), model.storage_nd_pd_reml)
    # (Φ'⊗X̃)'(Λ⊗D + Ind)⁻¹vec(ỸΦ)
    copyto!(model.storage_nd_2_reml, model.ỸΦ_reml)
    model.storage_nd_2_reml .= model.storage_nd_1_reml .* model.storage_nd_2_reml
    mul!(model.storage_pd_reml, transpose(model.storage_nd_pd_reml), model.storage_nd_2_reml)
    # Cholesky solve
    _, info = LAPACK.potrf!('U', G)
    info > 0 && throw("Gram matrix (Φ'⊗X̃)'(Λ⊗D + Ind)⁻¹(Φ'⊗X̃) is singular")
    LAPACK.potrs!('U', G, model.storage_pd_reml)
    copyto!(model.B_reml, model.storage_pd_reml)
    model.B_reml
end

function fisher_B!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    fill!(model.storage_nd_pd, zero(T))
    kron_axpy!(transpose(model.Φ), model.X̃, model.storage_nd_pd)
    fill!(model.storage_nd_1, zero(T))
    kron_axpy!(model.Λ, model.D, model.storage_nd_1)
    @inbounds @simd for i in eachindex(model.storage_nd_1)
        model.storage_nd_1[i] = one(T) / sqrt(model.storage_nd_1[i] + one(T))
    end
    lmul!(Diagonal(model.storage_nd_1), model.storage_nd_pd)
    mul!(model.storage_pd_pd, transpose(model.storage_nd_pd), model.storage_nd_pd)
    copyto!(model.Bcov, pinv(model.storage_pd_pd))
end

function fisher_B_reml!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    fill!(model.storage_nd_pd_reml, zero(T))
    kron_axpy!(transpose(model.Φ), model.X̃_reml, model.storage_nd_pd_reml)
    fill!(model.storage_nd_1_reml, zero(T))
    kron_axpy!(model.Λ, model.D_reml, model.storage_nd_1_reml)
    @inbounds @simd for i in eachindex(model.storage_nd_1_reml)
        model.storage_nd_1_reml[i] = one(T) / sqrt(model.storage_nd_1_reml[i] + one(T))
    end
    lmul!(Diagonal(model.storage_nd_1_reml), model.storage_nd_pd_reml)
    mul!(model.storage_pd_pd_reml, transpose(model.storage_nd_pd_reml), model.storage_nd_pd_reml)
    copyto!(model.Bcov_reml, pinv(model.storage_pd_pd_reml))
end

function fisher_Σ!(
    model :: MRTVCModel{T}
    ) where T <: BlasReal
    n, d = size(model.Ỹ, 1), size(model.Ỹ, 2)
    Fisher = zeros(T, 2d^2, 2d^2)
    W = zeros(T, d, d)
    Φ2 = kron(model.Φ, model.Φ)
    # (1, 1) block
    @inbounds for j in 1:d, i in j:d
        λi, λj = model.Λ[i], model.Λ[j]
        @simd for k in 1:n
            W[i, j] += model.D[k]^2 / (λi * model.D[k] + one(T)) / (λj * model.D[k] + one(T))
        end
        W[i, j] /= 2
    end
    LinearAlgebra.copytri!(W, 'L')
    mul!(view(Fisher, 1:d^2, 1:d^2), Φ2 * Diagonal(vec(W)), transpose(Φ2))
    # (2, 1) block
    fill!(W, zero(T))
    @inbounds for j in 1:d, i in j:d
        λi, λj = model.Λ[i], model.Λ[j]
        @simd for k in 1:n
            W[i, j] += model.D[k] / (λi * model.D[k] + one(T)) / (λj * model.D[k] + one(T))
        end
        W[i, j] /= 2
    end
    LinearAlgebra.copytri!(W, 'L')
    mul!(view(Fisher, (d^2 + 1):(2d^2), 1:d^2), Φ2 * Diagonal(vec(W)), transpose(Φ2))
    # (2, 2) block
    fill!(W, zero(T))
    @inbounds for j in 1:d, i in j:d
        λi, λj = model.Λ[i], model.Λ[j]
        @simd for k in 1:n
            W[i, j] += one(T) / (λi * model.D[k] + one(T)) / (λj * model.D[k] + one(T))
        end
        W[i, j] /= 2
    end
    LinearAlgebra.copytri!(W, 'L')
    mul!(view(Fisher, (d^2 + 1):(2d^2), (d^2 + 1):(2d^2)), Φ2 * Diagonal(vec(W)), transpose(Φ2))
    LinearAlgebra.copytri!(Fisher, 'L')
    vechFisher = zeros(T, (2 * d * (d + 1)) >> 1, (2 * d * (d + 1)) >> 1)
    D = duplication(d)
    for i in 1:2
        idx1 = Int(d * (d + 1) / 2 * (i - 1) + 1)
        idx2 = Int(d * (d + 1) / 2 * i)
        idx5, idx6 = d^2 * (i - 1) + 1, d^2 * i
        for j in i:2
            idx3 = Int(d * (d + 1) / 2 * (j - 1) + 1)
            idx4 = Int(d * (d + 1) / 2 * j)
            idx7, idx8 = d^2 * (j - 1) + 1, d^2 * j
            vechFisher[idx1:idx2, idx3:idx4] = D' * Fisher[idx5:idx6, idx7:idx8] * D
        end
    end
    copytri!(vechFisher, 'U')
    copyto!(model.Σcov, pinv(vechFisher))
end