"""
__MRVCModels__ stands for __M__ultivariate __R__esponse __V__ariance __C__omponents
linear mixed __Models__. `MRVCModels.jl` permits maximum likelihood (ML) or residual
maximum likelihood (REML) estimation and inference.
"""
module MultiResponseVarianceComponentModels

using IterativeSolvers
using LinearAlgebra
using Distributions
using SweepOperator
using InvertedIndices

using LinearAlgebra: BlasReal, BlasInt, copytri!, tril!, checksquare
using LinearAlgebra.BLAS: @blasfunc

const liblapack = LinearAlgebra.libblastrampoline

# exports
export VCModel,
    MultiResponseVarianceComponentModel,
    MRVCModel,
    # MRTVCModel,  # Deprecated: K=2 handled via dispatch in MRVCModel
    # fit.jl or simplefit.jl
    fit!,
    loglikelihood!,
    loglikelihood,
    # loglikelihood_reml,  # TODO: Add with REML support
    update_res!,
    update_Ω!,
    update_B!,
    # update_B_reml!,  # TODO: Add with REML support
    fisher_B!,
    # fisher_B_reml!,  # TODO: Add with REML support
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

include("multivariate_calculus.jl")
include("utils.jl")
include("variancecomponents/methods.jl")
include("variancecomponents/unstructured.jl")
include("variancecomponents/lowrankplusdiagonal.jl")
include("variancecomponents/omega.jl")              # NEW: ΩMatrix type
include("model.jl")               # NEW: Replaces MRVCModel.jl and SimpleMRVCModel.jl
# include("MRVCModel.jl")         # OLD: Deprecated
# include("MRTVCModel.jl")        # OLD: Deprecated, K=2 handled in model.jl
include("reml.jl")
# include("fit.jl")               # OLD: For old MRVCModel
include("simplefit.jl")           # Rewritten for new MRVCModel
# include("eigen.jl")             # OLD: For MRTVCModel only, TODO: add K=2 methods
# include("manopt.jl")            # OLD: Manifold optimization (performance issues)
include("parse.jl")
include("missing.jl")

end