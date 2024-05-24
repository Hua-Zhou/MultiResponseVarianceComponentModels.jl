import Manifolds: vector_transport_to!
struct IdentityTransport <: AbstractVectorTransportMethod end
vector_transport_to!(md::Stiefel, Y, p, X, q, ::IdentityTransport) = 
        project!(md, Y, q, X)

"""
    update_Σk!(model, k, rk)

Update the `model.Σ[k]` assuming it has rank `rk < d`, assuming inverse of 
covariance matrix `model.Ω` is available at `model.storage_nd_nd` and 
`model.Ω⁻¹R` precomputed.
"""
function update_Σk!(
    model :: MRVCModel{T},
    k     :: Integer,
    rk    :: Integer
    ) where T <: BlasReal
    d = size(model.Y, 2)
    (rk > d) && (return update_Σk!(model, k, Val(:MM)))
    Ω⁻¹ = model.storage_nd_nd
    # M = storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    M = model.storage_d_d_1
    # N = storage_d_d_2 = Σ[k] * (R' * V[k] * R) * Σ[k]
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    mul!(model.storage_d_d_3, model.storage_d_d_2, model.Σ[k])
    mul!(model.storage_d_d_2, model.Σ[k], model.storage_d_d_3)
    N = model.storage_d_d_2
    # minimize 2dot(sqrt.(diag(U'MU)), sqrt.(diag(U'NU))) by manifold optimization
    storage_d_r_1 = view(model.storage_d_d_3, :, 1:rk)
    storage_d_r_2 = view(model.storage_d_d_4, :, 1:rk)
    storage_d_r_3 = view(model.storage_d_d_5, :, 1:rk)
    storage_d_r_4 = view(model.storage_d_d_6, :, 1:rk)
    Usol          = view(model.storage_d_d_7, :, 1:rk)
    md = Stiefel(d, rk)
    # objective evaluator
    function cost(::Stiefel, U::Matrix{T})
        f = zero(T)
        mul!(storage_d_r_1, M, U)
        mul!(storage_d_r_2, N, U)
        @inbounds for j in 1:rk
            ajj = dot(view(U, :, j), storage_d_r_1[:, j])
            bjj = dot(view(U, :, j), storage_d_r_2[:, j])
            f  += 2sqrt(max(ajj * bjj, zero(T)))
        end
        f
    end
    # gradient evaluation
    function gradf!(md, rgrad, U)
        mul!(storage_d_r_1, M, U)
        mul!(storage_d_r_2, N, U)
        egrad = storage_d_r_3
        @inbounds for j in 1:rk
            ajj = dot(view(U, :, j), storage_d_r_1[:, j])
            bjj = dot(view(U, :, j), storage_d_r_2[:, j])
            egrad[:, j] .= 2 .* (sqrt(max(bjj / ajj, zero(T))) .* storage_d_r_1[:, j] .+ 
                sqrt(max(ajj / bjj, zero(T))) .* storage_d_r_2[:, j])
        end
        project!(md, rgrad, U, egrad)
        rgrad
    end
    # start point, exit with solution
    # U0 = diagm(d, rk, ones(T, rk))
    U0 = view(eigen!(Symmetric(model.Σ[k])).vectors, :, 1:rk)
    # U0 = random_point(md)
    Usol .= quasi_Newton(
        md,
        cost,
        gradf!,
        U0,
        memory_size = 32,
        cautious_update = true,
        evaluation = MutatingEvaluation(),
        vector_transport_method = IdentityTransport(),
        stopping_criterion = StopWhenGradientNormLess(1e-4),
        # debug=[:Iteration, " ", :Cost, "\n", 1, :Stop]
    )
    # determine optimal eigenvalues
    mul!(storage_d_r_1, M, Usol)
    mul!(storage_d_r_2, N, Usol)
    @inbounds for j in 1:rk
        ajj = dot(Usol[:, j], storage_d_r_1[:, j])
        bjj = dot(Usol[:, j], storage_d_r_2[:, j])
        σj = sqrt(max(bjj / ajj, zero(T)))
        if σj > 0
            Usol[:, j] .*= sqrt(σj)
        else
            Usol[:, j] .= 0
        end
    end
    mul!(model.Σ[k], Usol, transpose(Usol))
    model.Σ[k]
end

# """
#     update_Γk!(model, k, rk)

# Update the `model.Γ[k]` assuming it has rank `rk < d`, assuming inverse of 
# covariance matrix `model.Ω` is available at `model.storage_nd_nd` and 
# `model.Ω⁻¹R` precomputed.
# """
# function update_Γk!(
#     model :: MRVCModel{T},
#     k     :: Integer
#     ) where T <: BlasReal
#     d, rk = size(model.Γ[k])
#     Ω⁻¹ = model.storage_nd_nd
#     # M = storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
#     kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
#     M = Symmetric(model.storage_d_d_1)
#     ◺M = sqrt(M; rtol=1e-4)

#     # Store ΓₖΓₖᵀ in Σₖ
#     BLAS.syrk!('L', 'N', one(T), model.Γ[k], zero(T), model.Σ[k])
#     copytri!(model.Σ[k], 'L')
#     G = model.Σ[k]
#     # N = storage_d_d_2 = R' * V[k] * R
#     # C = Σ[k] * N * Σ[k]
#     mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
#     mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
#     N = Symmetric(model.storage_d_d_2)
#     ◺N = sqrt(N; rtol=1e-4)

#     C = Symmetric(model.Σ[k] * N * model.Σ[k])

#     Σ_new = (G * ◺N) / (◺M)
#     eig_Σ = eigen(Symmetric(Σ_new + transpose(Σ_new) / 2); sortby = x->-abs(x))
#     copyto!(model.Γ[k], view(eig_Σ.vectors, :, 1:rk))
#     for j in 1:rk
#         u = view(model.Γ[k], :, j)
#         σ = abs(dot(u, C, u) / dot(u, M, u))^(1//4)
#         for i in 1:d
#             model.Γ[k][i, j] *= σ
#         end
#     end

#     # update Σ[k]
#     mul!(model.Σ[k], model.Γ[k], transpose(model.Γ[k]))
#     model.Σ[k]
# end

"""
    update_Γk!(model, k)

Update the parameters `model.Γ[k]` and `model.Ψ[:,k]` assuming a diagonal plus
low rank structure for `model.Σ[k]`. Assumes covariance matrix `model.Ω` is 
available at `model.storage_nd_nd` and `model.Ω⁻¹R` precomputed.
"""
function update_Γk!(
    model :: MRVCModel{T},
    k     :: Integer
    ) where T <: BlasReal
    d, rk = size(model.Γ[k])
    Ω⁻¹ = model.storage_nd_nd
    # M = storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
    kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
    M = Symmetric(model.storage_d_d_1)

    # N = storage_d_d_2 = R' * V[k] * R
    mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
    mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
    N = Symmetric(model.storage_d_d_2)

    # Update ψ
    ψ = view(model.Ψ, :, k)
    for i in 1:d
        ψ[i] *= sqrt(N[i, i] / M[i, i])
    end

    # Store ΓₖΓₖᵀ + Ψₖ in Σₖ
    BLAS.syrk!('L', 'N', one(T), model.Γ[k], zero(T), model.Σ[k])
    copytri!(model.Σ[k], 'L')
    for i in 1:d
        model.Σ[k][i,i] += ψ[i]
    end

    # C = Σ[k] * N * Σ[k]
    C = Symmetric(model.Σ[k] * N * model.Σ[k])

    # Update Hₖ = Ψ^(-1/2) Γₖ
    H = view(model.storage_d_d_3, :, 1:rk)
    copyto!(H, model.Γ[k])
    for j in 1:rk
        for i in 1:d
            H[i, j] /= sqrt(ψ[i])
        end
    end

    sqrt_Ψ = Diagonal(sqrt.(ψ))
    M̃ = sqrt_Ψ * M * sqrt_Ψ
    C̃ = sqrt_Ψ \ (C / sqrt_Ψ)
    W = Symmetric(transpose(H) * H + I)

    # Solving Sylvester Equation AX + XB = C
    RHS = C̃ * (H / W)
    A = M̃
    B = W \ ((transpose(H) * C̃ * H) / W)
    Hsol = sylvester(A, B, RHS)

    # Update Γ[k]
    model.Γ[k] .= sqrt_Ψ * Hsol

    # update Σ[k]
    mul!(model.Σ[k], model.Γ[k], transpose(model.Γ[k]))
    for i in 1:d
        model.Σ[k][i,i] += ψ[i]
    end
    model.Σ[k]
end

# """
#     update_Γk!(manif, model, k)

# Update the `model.Σ[k]` assuming it has rank `rk < d`, assuming inverse of 
# covariance matrix `model.Ω` is available at `model.storage_nd_nd` and 
# `model.Ω⁻¹R` precomputed.
# """
# function update_Γk!(
#     manif :: AbstractManifold,
#     model :: MRVCModel{T},
#     k     :: Integer;
#     maxiter :: Integer = 100
#     ) where T <: BlasReal
#     d, rk = size(model.Γ[k])
#     Ω⁻¹ = model.storage_nd_nd
#     # M = storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
#     kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
#     M = Symmetric(model.storage_d_d_1)

#     # Store ΓₖΓₖᵀ in Σₖ
#     BLAS.syrk!('L', 'N', one(T), model.Γ[k], zero(T), model.Σ[k])
#     copytri!(model.Σ[k], 'L')

#     # N = storage_d_d_2 = R' * V[k] * R
#     # C = Σ[k] * N * Σ[k]
#     mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
#     mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
#     N = Symmetric(model.storage_d_d_2)

#     C = Symmetric(model.Σ[k] * N * model.Σ[k])

#     # minimize 2dot(sqrt.(diag(U'MU)), sqrt.(diag(U'NU))) by manifold optimization
#     storage_d_r_1 = view(model.storage_d_d_3, :, 1:rk)
#     storage_d_r_2 = view(model.storage_d_d_4, :, 1:rk)
#     storage_d_r_3 = view(model.storage_d_d_5, :, 1:rk)
#     storage_d_r_4 = view(model.storage_d_d_6, :, 1:rk)
#     Usol          = view(model.storage_d_d_7, :, 1:rk)

#     # objective evaluator
#     # convention is minimization, so optimize negative surrogate
#     function objective(::AbstractManifold, U::AbstractMatrix{T})
#         mul!(storage_d_r_1, C, U)
#         mul!(storage_d_r_2, M, U)
#         res = zero(T)
#         @inbounds for j in axes(U, 2)
#             u = view(U, :, j)
#             utCu = dot(u, view(storage_d_r_1, :, j))
#             utMu = dot(u, view(storage_d_r_2, :, j))
#             res += sqrt(abs(utMu * utCu))
#         end
#         return res
#     end
#     # gradient evaluation
#     # Euclidean gradient
#     function egrad!(
#         dU::AbstractMatrix{T}, 
#         U::AbstractMatrix{T})
#         d, r = size(dU)
#         mul!(storage_d_r_1, C, U)
#         mul!(storage_d_r_2, M, U)
#         fill!(dU, zero(T))
#         for j in 1:r
#             u    = view(U, :, j)
#             utCu = dot(u, view(storage_d_r_1, :, j))
#             utMu = dot(u, view(storage_d_r_2, :, j))
#             σ²   = sqrt(utCu / utMu)
#             for i in 1:d
#                 dU[i, j] = storage_d_r_1[i, j] / σ² + storage_d_r_2[i, j] * σ²
#             end
#         end
#         return dU
#     end
#     # Riemannian gradient is simply the projection of the Euclidean gradient dU
#     # at the point U
#     function rgrad!(
#         ::AbstractManifold, 
#         rgrad::AbstractMatrix{T}, 
#         U::AbstractMatrix{T})
#         egrad = storage_d_r_3
#         egrad!(egrad, U)
#         project!(manif, rgrad, U, egrad)
#     end
#     # start point, exit with solution
#     U0 = qr(model.Γ[k]).Q[:, 1:rk] # TODO: add storage to struct
#     copyto!(Usol, U0)
#     approx_hessg = ApproxHessianFiniteDifference(manif, Usol, rgrad!;
#                                                  steplength = 1e-4,
#                                                  evaluation = MutatingEvaluation(),
#                                                  retraction_method = PolarRetraction())
#     trust_regions!(
#         manif, 
#         objective, 
#         rgrad!,
#         approx_hessg, 
#         U0;
#         max_trust_region_radius=1.0,
#         evaluation = MutatingEvaluation(),
#         retraction_method = PolarRetraction(),
#         debug=[:Iteration, :Cost, ", ", DebugGradientNorm(), 1, :Stop, "\n"],
#         stopping_criterion = StopWhenAny(
#             StopAfterIteration(maxiter), 
#             StopWhenGradientNormLess(1e-8)
#         )
#     )
#     copyto!(Usol, U0)
#     copyto!(model.Γ[k], U0)
#     # determine optimal eigenvalues
#     mul!(storage_d_r_1, M, Usol)
#     mul!(storage_d_r_2, N, Usol)
#     @inbounds for j in 1:rk
#         u    = view(Usol, :, j)
#         utCu = dot(u, view(storage_d_r_1, :, j))
#         utMu = dot(u, view(storage_d_r_2, :, j))
#         σ    = (max(utCu / utMu, zero(T))) ^ (1//4)
#         for i in 1:d
#             model.Γ[k][i, j] *= σ
#         end
#     end
#     # Update Σ
#     mul!(model.Σ[k], model.Γ[k], transpose(model.Γ[k]))
#     model.Σ[k]
# end

# function update_Γk!(
#     manif :: FixedRankMatrices,
#     model :: MRVCModel{T},
#     k     :: Integer;
#     maxiter :: Integer = 200
#     ) where T <: BlasReal
#     d, rk = size(model.Γ[k])
#     Ω⁻¹ = model.storage_nd_nd
#     # M = storage_d_d_1 = gradient of tr(Ω⁻¹ (Σ[k] ⊗ V[k]))
#     kron_reduction!(Ω⁻¹, model.V[k], model.storage_d_d_1, true)
#     M = Symmetric(model.storage_d_d_1)

#     # Store ΓₖΓₖᵀ in Σₖ
#     BLAS.syrk!('L', 'N', one(T), model.Γ[k], zero(T), model.Σ[k])
#     copytri!(model.Σ[k], 'L')

#     # N = storage_d_d_2 = R' * V[k] * R
#     # C = Σ[k] * N * Σ[k]
#     mul!(model.storage_n_d, model.V[k], model.Ω⁻¹R)
#     mul!(model.storage_d_d_2, transpose(model.Ω⁻¹R), model.storage_n_d)
#     N = Symmetric(model.storage_d_d_2)

#     C = Symmetric(model.Σ[k] * N * model.Σ[k])

#     # minimize 2dot(sqrt.(diag(U'MU)), sqrt.(diag(U'NU))) by manifold optimization
#     storage_d_r_1 = view(model.storage_d_d_3, :, 1:rk)
#     storage_d_r_2 = view(model.storage_d_d_4, :, 1:rk)
#     storage_d_r_3 = view(model.storage_d_d_5, :, 1:rk)
#     storage_d_r_4 = view(model.storage_d_d_6, :, 1:rk)
#     Γsol          = view(model.storage_d_d_7, :, 1:rk)

#     # objective evaluator
#     # convention is minimization, so optimize negative surrogate
#     function objective(
#         manif::FixedRankMatrices,
#         Γ::AbstractMatrix{T})
#         Γ_pinv = pinv(Γ; atol=1e-4)
#         tr(transpose(Γ) * M * Γ) / 2 + tr(Γ_pinv * C * transpose(Γ_pinv)) / 2
#     end
#     function objective(
#         manif::FixedRankMatrices,
#         P::SVDMPoint)
#         # TODO: Better implementation of this
#         Γ = embed(manif, P)
#         Γ_pinv = pinv(Γ; atol=1e-4)
#         tr(transpose(Γ) * M * Γ) / 2 + tr(Γ_pinv * C * transpose(Γ_pinv)) / 2
#     end
#     # gradient evaluation
#     # Euclidean gradient
#     function egrad!(
#         dΓ::AbstractMatrix{T},
#         Γ::AbstractMatrix{T}, 
#         C::AbstractMatrix{T}, 
#         M::AbstractMatrix{T})
#         Γ_pinv = pinv(Γ; atol = 1e-4)
#         dΓ .= M * Γ - transpose(Γ_pinv) * Γ_pinv * C * transpose(Γ_pinv) + (I - Γ * Γ_pinv) * C * transpose(Γ_pinv) * Γ_pinv * transpose(Γ_pinv)
#     end
#     # Riemannian gradient is simply the projection of the Euclidean gradient dU
#     # at the point U
#     function rgrad!(manif::FixedRankMatrices, Y::UMVTVector, X::AbstractMatrix{T})
#         egrad = similar(X)
#         egrad!(egrad, X, C, M)
#         P = SVDMPoint(X)
#         project!(manif, Y, P, egrad)
#     end
    
#     function rgrad!(manif::FixedRankMatrices, Y::UMVTVector, X::SVDMPoint)
#         P = embed(manif, X)
#         egrad = similar(P)
#         egrad!(egrad, P, C, M)
#         project!(manif, Y, X, egrad)
#     end
#     function rgrad(manif::FixedRankMatrices, X::AbstractMatrix{T})
#         egrad = similar(X)
#         egrad!(egrad, X, C, M)
#         P = SVDMPoint(X)
#         project(manif, P, egrad)
#     end
    
#     function rgrad(manif::FixedRankMatrices, X::SVDMPoint)
#         P = embed(manif, X)
#         egrad = similar(P)
#         egrad!(egrad, P, C, M)
#         project(manif, X, egrad)
#     end
#     # start point, exit with solution
#     Γ0 = SVDMPoint(model.Γ[k])
#     conjugate_gradient_descent!(
#         manif, objective, rgrad, Γ0;
#         stepsize = ArmijoLinesearch(
#             manif; 
#             contraction_factor  = 0.95, 
#             sufficient_decrease = 0.05,
#             linesearch_stopsize = 1e-6),
#         retraction_method = PolarRetraction(),
#         debug = [:Iteration, :Cost, ", ", DebugGradientNorm(), 10, ", ", DebugStepsize(), :Stop, "\n"],
#         stopping_criterion = StopWhenAny(
#             StopAfterIteration(200), 
#             StopWhenGradientNormLess(1e-6),
#             StopWhenStepsizeLess(1e-6)
#         )
#     )
#     embed!(manif, Γsol, Γ0)
#     copyto!(model.Γ[k], Γsol)
#     # Update Σ
#     mul!(model.Σ[k], model.Γ[k], transpose(model.Γ[k]))
#     model.Σ[k]
# end