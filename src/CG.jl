function Ω_mul_x!(
    res     :: AbstractVector{T},
    model   :: MultiResponseVarianceComponentModel{T},
    x       :: AbstractArray{T},
    α       :: T,
    β       :: T
    ) where {T<:BlasReal}
    Xmat = model.storage_n_d_2
    copyto!(Xmat, x)
    if iszero(β) == true
        fill!(model.storage_n_d_1, zero(T))
    else
        lmul!(β, model.storage_n_d_1)
    end
    @inbounds for j in eachindex(model.V)
        BLAS.symm!('L', 'L', α, model.V[j], Xmat, zero(T), model.storage_n_d_3)
        BLAS.symm!('R', 'L', one(T), model.VarComp[j].Σ, model.storage_n_d_3, one(T), model.storage_n_d_1)
    end
    copyto!(res, model.storage_n_d_1)
end

function Ω_mul_x!(
    res     :: AbstractVector{T},
    model   :: MultiResponseVarianceComponentModel{T},
    x       :: AbstractArray{T}
    ) where {T<:BlasReal}
    Ω_mul_x!(res, model, x, one(T), zero(T))
end

"""
    ConjGrad!()

"""

# Want the solution of Ωx = vec(R)
# implicitly assume x0 = 0
function ConjGrad!(
    model   :: MultiResponseVarianceComponentModel{T};
    # x0      :: AbstractArray{T};
    tol     :: T = T(1e-8),
    maxiter :: Int = 100
    ) where {T<:BlasReal}
    # Assume Y - XB has already been stored in model.R
    rhs   = model.R
    x     = model.Ω⁻¹R

    # Initialize residual and initial direction
    resid = model.storage_nd_1
    copyto!(resid, rhs)
    axpy!(-one(T), x, resid)

    dir   = model.storage_nd_2
    copyto!(dir, -resid)

    iter = 0
    flag = false
    while iter ≤ maxiter
        iter += 1
        Ω_mul_x!(model.storage_nd_3, model, dir)
        dir_t_A_dir = dot(dir, model.storage_nd_3)
        β = -dot(resid, dir) / dir_t_A_dir
        # x .= x .+ β .* dir
        axpy!(β, dir, x)
        # update residual
        Ω_mul_x!(resid, model, x)
        axpy!(-one(T), rhs, resid)

        # Convergence Check
        if norm(resid, 2) < tol
            break
        else
            χ = dot(resid, model.storage_nd_3) / dir_t_A_dir
            # dir .= resid .+ χ .* dir
            axpby!(one(T), resid, χ, dir)
            if iter == maxiter
                @show norm(resid)
                flag = true
            end
        end
    end
    # flag == true && @warn "Conjugate Gradient Iterations did not converge"
end
