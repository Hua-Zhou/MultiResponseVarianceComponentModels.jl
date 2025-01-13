"""
    update_Σk_miss!(model::MRVCModel, k, Val(:MM))

MM update the `model.Σ[k]`, assuming inverse of covariance matrix `model.Ω` is available at
`model.storage_nd_nd`, `model.Ω⁻¹R` is precomputed, and Ω⁻¹P'CPΩ⁻¹ is available at
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
    loglikelihood_miss!(model::MRVCModel)

Return the value of surrogate Q-function of log-likelihood, overwrite `model.storage_nd_nd`
by inverse of covariance matrix `model.Ω`, `model.Y` by imputed response, and
`model.storage_nd_nd_miss` by Ω⁻¹P'CPΩ⁻¹. Assume `model.Ω` and `model.R` are already updated
according to `model.Σ` and `model.B`.
"""
function loglikelihood_miss!(
    model :: MRVCModel{T}
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
    # compute Ω⁻¹P'CPΩ⁻¹
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
    # compute imputed response
    nd = size(Ω⁻¹, 1)
    n_obs = nd - model.n_miss
    mul!(model.R, model.X, model.B)
    copyto!(model.storage_nd_1, model.R)
    model.storage_nd_2 .= @view model.storage_nd_1[model.P] # expected mean
    copy!(model.storage_n_obs, view(model.storage_nd_2, 1:n_obs))
    copy!(model.storage_n_miss, view(model.storage_nd_2, (n_obs + 1):nd))
    model.storage_n_obs .= model.Y_obs - model.storage_n_obs
    BLAS.gemv!('N', one(T), model.storage_n_miss_n_obs_1, model.storage_n_obs, one(T),
        model.storage_n_miss) # conditional mean
    copyto!(model.storage_nd_2, model.Y_obs)
    copy!(view(model.storage_nd_2, (n_obs + 1):nd), model.storage_n_miss)
    model.storage_nd_1 .= @view model.storage_nd_2[model.invP]
    copyto!(model.Y, model.storage_nd_1)
    logl /= -2
end

# function loglikelihood_miss!(
#     model :: MRVCModel{T}
#     ) where T <: BlasReal
#     nd = size(model.Ω, 1)
#     n_obs = nd - model.n_miss
#     PΩPt = model.storage_nd_nd_miss
#     PΩPt .= @view model.Ω[model.P, model.P]
#     logdetΩ = zero(T) # for log-likelihood
#     @inbounds for k in 1:n_obs
#         logdetΩ += log(PΩPt[k, k])
#         sweep!(PΩPt, k)
#     end
#     copytri!(PΩPt, 'U')
#     copy!(model.storage_n_miss_n_obs_1,
#         view(PΩPt, (n_obs + 1):nd, 1:n_obs)) # for conditional mean
#     copy!(model.storage_n_miss_n_miss_2, model.storage_n_miss_n_miss_1)
#     copy!(model.storage_n_miss_n_miss_1,
#         view(PΩPt, (n_obs + 1):nd, (n_obs + 1):nd)) # conditional variance
#     C = model.storage_n_miss_n_miss_2
#     @inbounds for k in (n_obs + 1):nd
#         logdetΩ += log(PΩPt[k, k])
#         sweep!(PΩPt, k) # PΩ⁻¹P' from sweep operator
#     end
#     copytri!(PΩPt, 'U')
#     PΩPt .= -one(T) .* PΩPt
#     logl = dot(view(PΩPt, (n_obs + 1):nd, (n_obs + 1):nd), C) # tr(PΩ⁻¹P'C) for log-likelihood
#     model.storage_nd_nd .= @view PΩPt[model.invP, model.invP]
#     Ω⁻¹ = model.storage_nd_nd
#     # assemble pieces for surrogate function of log-likelihood
#     copyto!(model.storage_nd_1, model.R)
#     mul!(model.storage_nd_2, Ω⁻¹, model.storage_nd_1)
#     logl += dot(model.storage_nd_1, model.storage_nd_2) + nd * log(2π) + logdetΩ
#     # compute Ω⁻¹P'CPΩ⁻¹
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
#     # compute imputed response
#     mul!(model.R, model.X, model.B)
#     copyto!(model.storage_nd_1, model.R)
#     model.storage_nd_2 .= @view model.storage_nd_1[model.P] # expected mean
#     copy!(model.storage_n_obs, view(model.storage_nd_2, 1:n_obs))
#     copy!(model.storage_n_miss, view(model.storage_nd_2, (n_obs + 1):nd))
#     model.storage_n_obs .= model.Y_obs - model.storage_n_obs
#     BLAS.gemv!('N', one(T), model.storage_n_miss_n_obs_1, model.storage_n_obs, one(T),
#         model.storage_n_miss) # conditional mean
#     copyto!(model.storage_nd_2, model.Y_obs)
#     copy!(view(model.storage_nd_2, (n_obs + 1):nd), model.storage_n_miss)
#     model.storage_nd_1 .= @view model.storage_nd_2[model.invP]
#     copyto!(model.Y, model.storage_nd_1)
#     logl /= -2
# end

"""
    permute(Y::AbstractVecOrMat)

Return permutation `P` such that `vec(Y)[P]` rearranges `vec(Y)` with missing values
spliced after non-missing values. Also return inverse permutation `invP` such that
`vec(Y)[P][invP] = vec(Y)`.
"""
function permute(Y::AbstractMatrix{Union{Missing, T}}) where T <: BlasReal
    idxall = findall(ismissing, Y)
    Y_imputed = similar(Matrix{T}, size(Y))
    Y_imputed[Not(idxall)] = Y[Not(idxall)]
    for (i, col) in enumerate(eachcol(Y))
        Y_imputed[findall(ismissing, col), i] .= mean(skipmissing(col))
    end
    n, d = size(Y)
    P = zeros(Int, n * d)
    i1, j1 = 0, 1
    n_miss = length(idxall)
    for (iter, idx) in enumerate(idxall)
        i2, j2 = Tuple(idx)
        P[end - n_miss + iter] = (j2 - 1) * n + i2
        r = ((j1 - 1) * n + i1 + 2 - iter):((j2 - 1) * n + i2 - iter)
        if length(r) > 0
            P[r] = ((j1 - 1) * n + i1 + 1):((j2 - 1) * n + i2 - 1)
            i1, j1 = Tuple(idx)
        else
            i1, j1 = Tuple(idx)
            continue
        end
    end
    i2, j2 = n + 1, d
    r = ((j1 - 1) * n + i1 + 1 - n_miss):((j2 - 1) * n + i2 - n_miss - 1)
    P[r] = ((j1 - 1) * n + i1 + 1):((j2 - 1) * n + i2 - 1)
    P, invperm(P), n_miss, Y_imputed
end

permute(y::AbstractVector) = permute(reshape(y, length(y), 1))
