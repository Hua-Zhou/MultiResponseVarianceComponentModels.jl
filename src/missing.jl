"""
    permute(Y::AbstractVecOrMat)

Return permutation `P` such that `vec(Y)[P]` rearranges `vec(Y)` with missing values 
spliced after non-missing values. Also return inverse permutation `invP` such that
`vec(Y)[P][invP] = vec(Y)`.
"""
function permute(Y::AbstractMatrix)
    idxall = findall(ismissing, Y)
    n, d = size(Y)
    P = zeros(Int, n * d)
    i1, j1 = 0, 1
    nmissing = length(idxall)
    for (iter, idx) in enumerate(idxall)
        i2, j2 = Tuple(idx)
        P[end - nmissing + iter] = (j2 - 1) * n + i2
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
    r = ((j1 - 1) * n + i1 + 1 - nmissing):((j2 - 1) * n + i2 - nmissing - 1)
    P[r] = ((j1 - 1) * n + i1 + 1):((j2 - 1) * n + i2 - 1)
    P, invperm(P)
end

permute(y::AbstractVector) = permute(reshape(y, length(y), 1))