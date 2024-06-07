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