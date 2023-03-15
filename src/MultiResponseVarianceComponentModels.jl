module MultiResponseVarianceComponentModels

using IterativeSolvers, LinearAlgebra, Manopt, Manifolds, Distributions
import LinearAlgebra: BlasReal, copytri!
export fit!,
    kron_axpy!, 
    kron_reduction!, 
    loglikelihood!,
    MultiResponseVarianceComponentModel,
    update_res!,
    update_Ω!,
    fisher_Σ!,
    lrt,
    h2,
    rg

include("variancecomponents.jl")
include("multivariate_calculus.jl")
include("fit.jl")
include("parse.jl")
# include("manopt.jl")

end