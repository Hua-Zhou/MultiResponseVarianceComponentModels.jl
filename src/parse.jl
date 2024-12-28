"""
    lrt(model1::MRVCModel, model0::MRVCModel)

Perform a variation of the likelihood ratio test for univariate variance components models as in 
Molenberghs and Verbeke 2007 with model1 and model0 being the full and nested models, respectively.
"""
function lrt(
    model1 :: VCModel,
    model0 :: VCModel
    )
    df = length(model1.V) - length(model0.V)
    @assert df > 0
    @assert size(model0.Σ[1], 1) == 1
    @assert size(model1.Σ[1], 1) == 1
    lrt = 2 * (model1.logl[1] - model0.logl[1])
    coefs = [2.0^-df * binomial(df, i) for i in 1:df]
    ps = [ccdf(Chisq(i), lrt) for i in 1:df]
    sum(coefs .* ps)
end

"""
    h2(model::VCModel)

Calculate heritability estimates and their standard errors, assuming that all variance components 
capture genetic effects except the last term. Also return total heritability from sum of individual 
contributions and its standard error.
"""
function h2(model::VCModel)
    m, d = length(model.Σ), size(model.Σ[1], 1)
    h2s  = zeros(eltype(model.Y), m, d)
    ses  = zeros(eltype(model.Y), m, d)
    tot  = sum([model.Σ[l] for l in 1:m])
    idx  = findvar(d)
    s    = ◺(d)
    for j in 1:d
        for i in 1:m
            w = [idx[j] + s * (l - 1) for l in 1:m]
            Σcov = model.Σcov[w, w]
            if i != m
                h2s[i, j] = model.Σ[i][j, j] / tot[j, j]
                ∇h2       = [-model.Σ[i][j, j] / tot[j, j]^2 for l in 1:m]
                ∇h2[i]    = (tot[j, j] - model.Σ[i][j, j]) / tot[j, j]^2
                ses[i, j] = sqrt(∇h2' * Σcov * ∇h2)
            else
                h2s[i, j] = 1 - model.Σ[i][j, j] / tot[j, j]
                ∇h2       = ones(m) * model.Σ[i][j, j] / tot[j, j]^2
                ∇h2[i]    = -(tot[j, j] - model.Σ[i][j, j]) / tot[j, j]^2
                ses[i, j] = sqrt(∇h2' * Σcov * ∇h2)
            end
        end
    end
    h2s, ses
end

function findvar(d::Int)
    s, r = ◺(d), d 
    idx = ones(Int, d)
    for j in 2:length(idx)
        idx[j] = idx[j - 1] + r
        r -= 1
    end
    idx
end

"""
    rg(model::VCModel)

Calculate genetic/residual correlation estimates and their standard errors.
"""
function rg(model::VCModel)
    m, d = length(model.Σ), size(model.Σ[1], 1)
    @assert d > 1
    rgs = [zeros(eltype(model.Y), d, d) for _ in 1:m]
    ses = [ones(eltype(model.Y), d, d) for _ in 1:m]
    idx = findvar(d)
    s   = ◺(d)
    for i in 1:m
        D = Diagonal(model.Σ[i])
        for j in 1:d
            D[j, j] = 1 / sqrt(D[j, j])
        end
        rgs[i] = D * model.Σ[i] * D
        for j in 1:d
            w = idx .+ s * (i - 1)
            for k in (j + 1):d
                Σcov = model.Σcov[[(k - j) + w[j], w[j], w[k]], [(k - j) + w[j], w[j], w[k]]]
                ∇rg = [1 / sqrt(model.Σ[i][j, j] * model.Σ[i][k, k]),
                    -0.5 * model.Σ[i][j, k] / sqrt(model.Σ[i][k, k] * model.Σ[i][j, j]^3),
                    -0.5 * model.Σ[i][j, k] / sqrt(model.Σ[i][k, k]^3 * model.Σ[i][j, j])]
                ses[i][k, j] = sqrt(∇rg' * Σcov * ∇rg)
            end
        end
    end
    [copytri!(ses[i], 'L') for i in 1:m]
    rgs, ses
end