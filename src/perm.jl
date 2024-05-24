x = Matrix{Union{Float64, Missing}}(missing, 4, 3)
y = rand(4, 3)
copyto!(x, y)
any(ismissing, x)
ind = rand(1:length(x), 6)
x[ind] .= missing
any(ismissing, x)
"""
    permute(Y)

Return permutation `p` 
"""
function permute(Y)
    idxall = findall(ismissing, Y)
    n, d = size(Y)
    p = zeros(Int, n * d)
    i1, j1 = 0, 1
    nmissing = length(idxall)
    for (iter, idx) in enumerate(idxall)
        i2, j2 = Tuple(idx)
        p[end - nmissing + iter] = (j2 - 1) * n + i2
        r = ((j1 - 1) * n + i1 + 2 - iter):((j2 - 1) * n + i2 - iter)
        if length(r) > 0
            p[r] = ((j1 - 1) * n + i1 + 1):((j2 - 1) * n + i2 - 1)
            i1, j1 = Tuple(idx)
        else
            i1, j1 = Tuple(idx)
            continue
        end
    end
    i2, j2 = n + 1, d
    r = ((j1 - 1) * n + i1 + 1 - nmissing):((j2 - 1) * n + i2 - nmissing - 1)
    p[r] = ((j1 - 1) * n + i1 + 1):((j2 - 1) * n + i2 - 1)
    p, invperm(p)
end

p, invp = permute(x)

vec(x)[p][invp]
vec(x)

u .= @view
  v[p].

permute! check out
# https://julialang.org/blog/2016/02/iteration/
# https://discourse.julialang.org/t/fastest-way-to-permute-array-given-some-permutation/49687/11
# https://docs.julialang.org/en/v1/stdlib/SparseArrays/
# https://github.com/JuliaLang/julia/issues/35829
# https://robertsweeneyblanco.github.io/Programming_for_Mathematical_Applications/content/Sparse_Matrices/Sparse_Matrices_In_Julia.html
x[ind2]
typeof(ind2[1])
ind2[1]

x = collect(1:10)
splice!(x, 5, 21)


#permute!

x = reshape(Vector(1:16), (4,4))
p = [0 1 0 0; 0 0 0 1; 1 0 0 0; 0 0 1 0]
i = [2, 4, 1, 3]
p * x
x[i, :]
x * transpose(p)
x[:, i]
p * x * transpose(p)
x[i, i]
