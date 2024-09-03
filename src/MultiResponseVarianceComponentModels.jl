"""
__MRVCModels__ stands for __M__ultivariate __R__esponse __V__ariance __C__omponents
linear mixed __Models__. `MRVCModels.jl` permits maximum likelihood (ML) or residual
maximum likelihood (REML) estimation and inference.
"""
module MultiResponseVarianceComponentModels

using IterativeSolvers, LinearAlgebra, Manopt, Manifolds, Distributions, SweepOperator, InvertedIndices
import LinearAlgebra: BlasReal, copytri!
export VCModel,
    MultiResponseVarianceComponentModel,
    MRVCModel,
    MRTVCModel,
    # fit.jl or eigen.jl
    fit!,
    loglikelihood!,
    loglikelihood,
    loglikelihood_reml,
    update_res!,
    update_Ω!,
    update_B!,
    update_B_reml!,
    fisher_B!,
    fisher_B_reml!,
    fisher_Σ!,
    # missing.jl
    permute,
    # parse.jl
    lrt,
    h2,
    rg,
    # multivariate_calculus.jl
    kron_axpy!,
    kron_reduction!,
    vech,
    ◺,
    duplication,
    commutation

abstract type VCModel end

include("MRVCModel.jl")
include("MRTVCModel.jl")
include("multivariate_calculus.jl")
include("reml.jl")
include("fit.jl")
include("eigen.jl")
include("manopt.jl")
include("parse.jl")
include("missing.jl")

function Base.show(io::IO, model::VCModel)
    if model.reml
        n, d, p, m = size(model.Y_reml, 1), size(model.Y_reml, 2), size(model.X_reml, 2), length(model.V_reml)
    else
        n, d, p, m = size(model.Y, 1), size(model.Y, 2), size(model.X, 2), length(model.V)
    end
    if d == 1 && model isa MRTVCModel
        printstyled(io, "A univariate response two variance component model\n"; underline = true)
    elseif d == 1
        printstyled(io, "A univariate response variance component model\n"; underline = true)
    elseif d == 2 && model isa MRTVCModel
        printstyled(io, "A bivariate response two variance component model\n"; underline = true)
    elseif d == 2
        printstyled(io, "A bivariate response variance component model\n"; underline = true)
    elseif model isa MRTVCModel
        printstyled(io, "A multivariate response two variance component model\n"; underline = true)
    else
        printstyled(io, "A multivariate response variance component model\n"; underline = true)
    end
    print(io, "   * number of responses: ")
    printstyled(io, "$d\n"; color = :yellow)
    print(io, "   * number of observations: ")
    printstyled(io, "$n\n"; color = :yellow)
    print(io, "   * number of fixed effects: ")
    printstyled(io, "$p\n"; color = :yellow)
    print(io, "   * number of variance components: ")
    printstyled(io, "$m"; color = :yellow)
end


end