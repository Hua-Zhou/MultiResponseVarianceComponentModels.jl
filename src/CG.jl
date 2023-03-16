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
    model   :: MultiResponseVarianceComponentModel{T},
    x0      :: AbstractArray{T};
    tol     :: T = T(1e-8),
    maxiter :: Int = 100
    ) where {T<:BlasReal}
    # Assume Y - XB has already been stored in model.R
    x     = model.Ω⁻¹R
    # copyto!(x, model.R)
    copyto!(x, x0)

    # Initialize residual and initial direction
    resid = model.R
    axpy!(-one(T), x, resid)
    
    dir   = model.storage_nd_1
    copyto!(dir, model.R)
    
    rtr_old = sum(abs2, resid)
    rtr_new = rtr_old
    iter = 0
    while iter ≤ maxiter
        rtr_old = rtr_new
        Ω_mul_x!(model.storage_nd_2, model, dir)
        α = rtr_old / dot(dir, model.storage_nd_2)
        # x     .= x .+ α .* dir
        axpy!(α, dir, x)
        # resid .= resid .- α .* model.storage_nd_2
        axpy!(-α, model.storage_nd_2, resid)
        rtr_new = sum(abs2, resid)
        if sqrt(rtr_new) < tol
            # @show rtr_new
            # @show iter
            break
        else
            β = rtr_new / rtr_old
            axpby!(one(T), resid, β, dir)
            # dir .= resid .+ β .* dir
            iter += 1
        end
    end
end
