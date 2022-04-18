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
    model :: MultiResponseVarianceComponentModel{T},
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
