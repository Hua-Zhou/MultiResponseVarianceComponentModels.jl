

function project_null!(
    model::MRVCModel{T},
    ) where {T}
    K = length(model.Ω.VarComp)
    Y = model.Y
    X = model.X
    n, p = size(X)
    # initial fixed effect coefficients B by ordinary least squares (Cholesky solve)
    L = copyto!(model.storage_p_p, model.xtx)
    _, info = LAPACK.potrf!('L', model.storage_p_p)
    info > 0 && throw(PosDefException(info))
    if isempty(X)
        nothing
    else
        LAPACK.potrs!('L', L, copyto!(model.storage_p_d, model.xty))
        Yperp = copyto!(model.storage_n_d, Y)
        mul!(Yperp, X, model.storage_p_d, -one(T), one(T))
        copyto!(model.Y, Yperp)
        for i in 1:K
            Vi = model.Ω.VarComp[i].V
            B = X \ Vi
            Vperp = copyto!(model.Ω.VarComp[i].storage_nn, Vi)
            mul!(Vperp, X, B, -one(T), one(T))
            B = X \ transpose(Vperp)
            mul!(Vperp, transpose(B), transpose(X), -one(T), one(T))
            copyto!(model.Ω.VarComp[i].V, Vperp)
        end
    end
    return nothing
end