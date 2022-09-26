"""
    lrt(model1::MultiResponseVarianceComponentModel, model0::MultiResponseVarianceComponentModel)

Perform a variation of the likelihood ratio test as in Molenberghs and Verbeke 2007 with
model1 and model0 being the full and nested models, respectively.
"""
function lrt(
    model1 :: MultiResponseVarianceComponentModel,
    model0 :: MultiResponseVarianceComponentModel
    )
    df = length(model1.V) - length(model0.V)
    @assert df > 0
    lrt = 2 * (model1.logl[1] - model0.logl[1])
    coefs = [2.0^-df * binomial(df, i) for i in 1:df]
    ps = [ccdf(Chisq(i), lrt) for i in 1:df]
    sum(coefs .* ps)
end