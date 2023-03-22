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
# implicitly assume x0 is stored in model.Ω⁻¹R
function ConjGrad!(
    model   :: MultiResponseVarianceComponentModel{T};
    tol     :: T = T(1e-8),
    maxiter :: Int = 100
    ) where {T<:BlasReal}
    # Assume Y - XB has already been stored in model.R
    rhs   = model.R
    x     = model.Ω⁻¹R

    # Initialize residual and initial direction
    resid = model.storage_nd_1
    copyto!(resid, rhs)

    dir   = model.storage_nd_2
    copyto!(dir, resid)

    iter = 0
    rtr_old = sum(abs2, resid)
    while iter ≤ maxiter
        iter += 1
        Ω_mul_x!(model.storage_nd_3, model, dir)
        dir_t_A_dir = dot(dir, model.storage_nd_3)
        α = sum(abs2, resid) / dir_t_A_dir
        # Update x
        # x .= x .+ α .* dir
        axpy!(α, dir, x)
        # update residual
        # resid .= resid .- α .* Ω * dir
        axpy!(-α, model.storage_nd_3, resid)

        rtr_new = sum(abs2, resid)
        # @show norm(resid)
        # Convergence Check
        if sqrt(rtr_new) < tol
            break
        else
            β = rtr_new / rtr_old
            # dir .= resid .+ β .* dir
            axpby!(one(T), resid, β, dir)
            if iter == maxiter
                @show sqrt(rtr_new)
            end
        end
        rtr_old = rtr_new;
    end
end

# A function to avoid pointless allocation of a fixed matrix
function fill_RHS!(
    RHS::AbstractMatrix{T},
    model::MultiResponseVarianceComponentModel{T}
    ) where {T<:AbstractFloat}
    d    = size(model.Y, 2)
    n, p = size(model.X)
    fill!(RHS, zero(T))
    # RHS is nd × (pd + 1)
    copyto!(view(RHS, :, 1), model.Y)

    for k in 1:d
        startrowidx = (k - 1) * n
        startcolidx = (k - 1) * p + 1
        for j in 1:p
            for i in 1:n
                RHS[startrowidx + i, startcolidx + j] = model.X[i, j]
            end
        end
    end
    return RHS
end

# this is actually modified Gram-schmidt
function gramschmidt!(
    V::AbstractMatrix{T};
    evals::Int = 2
    ) where {T<:AbstractFloat}
    for j in axes(V, 2)
        if j == 1
            # u_1 is simply the normalized v_1
            normalize!(view(V, :, j))
        else
            # Double evaluation achieves minimum rounding error
            for _ in 1:evals
                for i in 1:j-1
                    vtu = dot(view(V, :, j), view(V, :, i))
                    axpy!(-vtu, view(V, :, i), view(V, :, j))
                end
                normalize!(view(V, :, j))
            end
        end
    end
end

# Method is from Ji and Li, 2015.
# "A breakdown-free block conjugate gradient method"
# Method can be improved by preconditioning and warm-starts
# Already works remarkably well without either
# Beauty of the algorithm is that makes computation
# of the Fisher Information practically free
function BlockConjGrad!(
    model      :: MultiResponseVarianceComponentModel{T},
    X          :: AbstractMatrix{T},
    Resid      :: AbstractMatrix{T},
    P          :: AbstractMatrix{T},
    Q          :: AbstractMatrix{T},
    Pt_Q       :: AbstractMatrix{T},
    𝒜          :: AbstractMatrix{T},
    ℬ          :: AbstractMatrix{T};
    maxiter    :: Int  = 100,
    tol        :: T    = T(1e-4),
    initialize :: Bool = true
    ) where {T<:AbstractFloat}
    # If initialize is true, ignore warm-start
    if initialize == true 
        fill!(X, zero(T))
        fill_RHS!(Resid, model)
    else
        fill_RHS!(Resid, model)
        for j in axes(X, 2)
            Ω_mul_x!(view(Resid, :, j), model, view(X, :, j), -one(T), one(T))
        end
    end

    copyto!(P, Resid)
    # Orthogonalize matrix P, 
    # Nice trick using Cholesky factor of Gram matrix
    # Not likely to be numerically stable
    # Gram-Schmidt is another option to iterate in-place
    gramschmidt!(P)
    iter = 0
    failure = false
    err = one(T)
    while iter ≤ maxiter
        iter += 1
        # BLAS.symm!('L', 'L', one(T), A, P, zero(T), Q)
        for j in axes(P, 2)
            Ω_mul_x!(view(Q, :, j), model, view(P, :, j))
        end
        mul!(Pt_Q, transpose(P), Q)

        _, info = LAPACK.potrf!('L', Pt_Q)
        info > 0 && error("Block Conjugate Gradient Failed on iter $iter !")
        mul!(𝒜, transpose(P), Resid)
        LAPACK.potrs!('L', Pt_Q, 𝒜)

        # BLAS.gemm!('N', 'N', one(T), P, 𝒜, one(T), X)
        mul!(X, P, 𝒜, one(T), one(T))
        # BLAS.gemm!('N', 'N', -one(T), Q, 𝒜, one(T), Resid)
        mul!(Resid, Q, 𝒜, -one(T), one(T))

        err = sqrt(sum(abs2, Resid))
        if err < tol
            # @show err
            # @show iter
            break
        elseif iter == maxiter
            failure = true
        else
            # BLAS.gemm!('T', 'N', -one(T), Q, Resid, zero(T), ℬ)
            mul!(ℬ, transpose(Q), Resid, -one(T), zero(T))
            LAPACK.potrs!('L', Pt_Q, ℬ)

            # BLAS.gemm!('N', 'N', one(T), P, ℬ, zero(T), Q)
            mul!(Q, P, ℬ)
            axpy!(one(T), Resid, Q)

            copyto!(P, Q)
            gramschmidt!(P)
        end
    end
    failure == true && throw("Block Conjugate Gradient Failed to Converge!")
    return X
end