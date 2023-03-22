
function update_RtVR!(
    model::MultiResponseVarianceComponentModel{T}
    ) where {T<:BlasReal}
    @inbounds for i in eachindex(model.V)
        # Turns out Hᵢ ≡ RᵀVᵢR
        BLAS.symm!('L', 'L', one(T), model.V[i], model.R, zero(T), model.storage_n_d_1)
        BLAS.gemm!('T', 'N', one(T), model.R, model.storage_n_d_1, zero(T), model.RtVR[i])
    end
end

# Updates the variance components by a projected coordinate descent update
function update_Σ_MoM!(
    model::MultiResponseVarianceComponentModel{T};
    maxiter::Int = 500,
    eps::T = zero(T),
    reltol::T = T(1e-6)
    ) where {T<:BlasReal}
    H         = model.storage_d_d_1
    Σ_new     = model.storage_d_d_2
    Σ_reldiff = model.storage_d_d_3
    iter      = 0
    failure   = false
    while iter ≤ maxiter
        iter += 1
        reldiff = zero(T)
        for j in eachindex(model.V)
            copyto!(H, model.RtVR[j])
            for i in eachindex(model.V)
                if i != j
                    axpy!(-model.V_sqnorm[i, j], model.VarComp[i].Σ, H)
                end
            end
            H ./= model.V_sqnorm[j, j]
            for i in axes(H, 1)
                model.storage_d[i] = H[i, i]
            end
            # Taking the nearest PD/Semidefinite matrix
            vals, vecs = LAPACK.syev!('V', 'L', H)
            for i in eachindex(vals)
                if vals[i] < eps
                    if j == 1
                        vals[i] = sqrt(eps)
                    else
                        vals[i] = zero(T)
                    end
                else
                    vals[i] = sqrt(vals[i])
                end
                lmul!(vals[i], view(vecs, :, i))
            end
            BLAS.syrk!('L', 'N', one(T), vecs, zero(T), Σ_new)
            copytri!(Σ_new, 'L')
            # Checking for convergence
            Σ_reldiff     .= abs.(Σ_new ./ model.VarComp[j].Σ) .- 1
            max_Σ_reldiff  = norm(Σ_reldiff)
            reldiff        = max(reldiff, max_Σ_reldiff)
            isnan(reldiff) && error("Maximum Relative Change is NaN!\n")
            copyto!(model.VarComp[j].Σ, Σ_new)
        end
        if reldiff < reltol 
            break
        elseif iter == maxiter
            failure = true
        end
        # Don't necessarily need to evaluate the objective function since we have convexity
        # update_Ω!(model)
    end
    failure == true && throw("Variance Components Update Failed to Converge!")
end

# Update variance components by projected gradient descent using a fixed stepsize
# Projected gradient descent is incredibly slow to converge
function update_Σ_MoM_grad!(
    model::MultiResponseVarianceComponentModel{T};
    maxiter::Int = 100,
    eps::T = zero(T),
    reltol::T = T(1e-6)
    ) where {T<:BlasReal}
    # Lipschitz constant
    L = eigmax(model.V_sqnorm)
    Σ_iter    = model.storages_d_d
    Σ_new     = model.storage_d_d_1
    grad      = model.storage_d_d_2
    Σ_reldiff = model.storage_d_d_3
    for iter in 1:maxiter
        reldiff = zero(T)
        # Gradient step
        for j in eachindex(model.V)
            copyto!(Σ_iter[j], model.VarComp[j].Σ)
            copyto!(grad, -model.RtVR[j])
            for i in eachindex(model.V)
                axpy!(model.V_sqnorm[i, j], model.VarComp[i].Σ, grad)
            end
            axpy!(-inv(L), grad, Σ_iter[j])
        end
        # Projection onto constraint set
        for j in eachindex(model.V)
            vals, vecs = LAPACK.syev!('V', 'L', Σ_iter[j])
            for i in eachindex(vals)
                if vals[i] < eps
                    vals[i] = eps
                else
                    vals[i] = sqrt(vals[i])
                end
                lmul!(vals[i], view(vecs, :, i))
            end
            BLAS.syrk!('L', 'N', one(T), vecs, zero(T), Σ_new)
            copytri!(Σ_new, 'L')
            # Checking for convergence
            Σ_reldiff .= abs.(Σ_new ./ model.VarComp[j].Σ) .- 1
            max_Σ_reldiff = norm(Σ_reldiff)
            reldiff = max(reldiff, max_Σ_reldiff)
            isnan(reldiff) && throw("Maximum Relative Change is NaN!\n")
            # Assigning output into Σ
            copyto!(model.VarComp[j].Σ, Σ_new)
        end
        if reldiff < reltol
            @show reldiff
            @show iter
            break
        elseif iter == maxiter
            @show reldiff
            @show iter
        end
    end
end