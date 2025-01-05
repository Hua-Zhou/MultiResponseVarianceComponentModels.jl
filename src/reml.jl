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

function loglikelihood_reml!(
    model :: MRVCModel{T}
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

function update_res_reml!(
    model :: MRVCModel{T}
    ) where T <: BlasReal
    # update R = Y - XB
    BLAS.gemm!('N', 'N', -one(T), model.X_reml, model.B_reml, one(T), copyto!(model.R_reml, model.Y_reml))
    model.R
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

function project_null(
    Y :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    V :: Vector{<:AbstractMatrix{T}}
    ) where {T <: Real}
    n, p, m = size(X, 1), size(X, 2), length(V)
    # basis of N(Xᵗ)
    Xᵗ = Matrix{T}(undef, p, n)
    transpose!(Xᵗ, X)
    A = nullspace(Xᵗ)
    s = size(A, 2) 
    Ỹ = transpose(A) * Y
    Ṽ = Vector{Matrix{T}}(undef, m)
    storage = zeros(n, s)
    for i in 1:m
        mul!(storage, V[i], A)
        Ṽ[i] = BLAS.gemm('T', 'N', A, storage)
    end
    Ỹ, Ṽ, A
end