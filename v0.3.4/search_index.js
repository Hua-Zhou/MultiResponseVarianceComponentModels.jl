var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"MultiResponseVarianceComponentModels\nMRVCModel\nMRTVCModel\nfit!","category":"page"},{"location":"api/#MultiResponseVarianceComponentModels","page":"API","title":"MultiResponseVarianceComponentModels","text":"MRVCModels stands for Multivariate Response Variance Components linear mixed Models. MRVCModels.jl permits maximum likelihood (ML) or residual maximum likelihood (REML) estimation and inference.\n\n\n\n\n\n","category":"module"},{"location":"api/#MultiResponseVarianceComponentModels.MRVCModel","page":"API","title":"MultiResponseVarianceComponentModels.MRVCModel","text":"MRVCModel(\n    Y::AbstractVecOrMat,\n    X::Union{Nothing, AbstractVecOrMat},\n    V::Union{AbstractMatrix, Vector{<:AbstractMatrix}}\n    )\n\nCreate a new MRVCModel instance from response matrix Y, predictor matrix X,  and kernel matrices V.\n\nKeyword arguments\n\nse::Bool        calculate standard errors; default true\nreml::Bool      pursue REML estimation instead of ML estimation; default false\n\nExtended help\n\nWhen there are two variance components, computation can be reduced by avoiding large matrix  inversion per iteration, which is achieved with MRTVCModel instance. MRTVCModels  stands for Multivariate Response Two Variance Components linear mixed Models. MRVCModel is more general, since it is not limited to two variance  components case. For MRTVCModel, the number of variance components must be two.\n\n\n\n\n\n","category":"type"},{"location":"api/#MultiResponseVarianceComponentModels.MRTVCModel","page":"API","title":"MultiResponseVarianceComponentModels.MRTVCModel","text":"MRTVCModel(\n    Y::AbstractVecOrMat,\n    X::Union{Nothing, AbstractVecOrMat},\n    V::Vector{<:AbstractMatrix}\n    )\n\nCreate a new MRTVCModel instance from response matrix Y, predictor matrix X,  and kernel matrices V. The number of variance components must be two.\n\nKeyword arguments\n\nse::Bool        calculate standard errors; default true\nreml::Bool      pursue REML estimation instead of ML estimation; default false\n\n\n\n\n\n","category":"type"},{"location":"api/#MultiResponseVarianceComponentModels.fit!","page":"API","title":"MultiResponseVarianceComponentModels.fit!","text":"fit!(model::MRVCModel)\nfit!(model::MRTVCModel)\n\nFit a multivariate response variance components model by MM or EM algorithm.\n\nKeyword arguments\n\nmaxiter::Int        maximum number of iterations; default 1000\nreltol::Real        relative tolerance for convergence; default 1e-6\nverbose::Bool       display algorithmic information; default true\ninit::Symbol        initialization strategy; :default initializes by least squares, while\n    :user uses user-supplied values at model.B and model.Σ\nalgo::Symbol        optimization algorithm; :MM (default) or :EM\nlog::Bool           record iterate history or not; default false\n\nExtended help\n\nMM algorithm is provably faster than EM algorithm in this setting, so recommend trying  MM algorithm first, which is the default algorithm, and switching to EM algorithm only if  there are  convergence issues.\n\n\n\n\n\n","category":"function"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#Simulate-data","page":"Examples","title":"Simulate data","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"using MultiResponseVarianceComponentModels, LinearAlgebra, Random\nRandom.seed!(6789)\nn = 1_000;  # n of observations\nd = 4;      # n of responses\np = 10;     # n of covariates\nm = 5;      # n of variance components\nX = rand(n, p);\nB = rand(p, d)\nV = [zeros(n, n) for _ in 1:m]; # kernel matrices\nΣ = [zeros(d, d) for _ in 1:m]; # variance components\nfor i in 1:m\n    Vi = randn(n, n)\n    copy!(V[i], Vi' * Vi)\n    Σi = randn(d, d)\n    copy!(Σ[i], Σi' * Σi)\nend\nΩ = zeros(n * d, n * d); # overall nd-by-nd covariance matrix Ω\nfor i = 1:m\n    Ω += kron(Σ[i], V[i])\nend\nΩchol = cholesky(Ω);\nY = X * B + reshape(Ωchol.L * randn(n * d), n, d)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"note: Note\nIn the case of heritability and genetic correlation analyses, one can use classic genetic relationship matrices (GRMs) for boldsymbolV_i's, which in turn can be constructed using SnpArrays.jl.","category":"page"},{"location":"examples/#Maximum-likelihood-estimation","page":"Examples","title":"Maximum likelihood estimation","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = MRVCModel(Y, X, V)\n@timev fit!(model)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Then variance components and mean effects estimates can be accessed through","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model.Σ\nmodel.B\nhcat(vec(B), vec(model.B))\nreduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:m])","category":"page"},{"location":"examples/#Standard-errors","page":"Examples","title":"Standard errors","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Sampling variance and covariance of these estimates are","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model.Σcov\nmodel.Bcov","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Corresponding standard error of these estimates are","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"sqrt.(diag(model.Σcov))\nsqrt.(diag(model.Bcov))","category":"page"},{"location":"examples/#Residual-maximum-likelihood-estimation","page":"Examples","title":"Residual maximum likelihood estimation","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"For REML estimation, you can instead:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = MRVCModel(Y, X, V; reml = true)\n@timev fit!(model)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Variance components and mean effects estimates and their standard errors can be accessed through:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model.Σ\nmodel.B_reml\nhcat(vec(B), vec(model.B_reml))\nreduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:m])\nmodel.Σcov\nmodel.Bcov_reml\nsqrt.(diag(model.Σcov))\nsqrt.(diag(model.Bcov_reml))","category":"page"},{"location":"examples/#Estimation-only","page":"Examples","title":"Estimation only","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Calculating standard errors can be memory-consuming, so you could instead forego such calculation via:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = MRVCModel(Y, X, V; se = false)\n@timev fit!(model)","category":"page"},{"location":"examples/#Special-case:-missing-response","page":"Examples","title":"Special case: missing response","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"You can also fit data with missing response. For example:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Y_miss = Matrix{Union{eltype(Y), Missing}}(missing, size(Y))\ncopy!(Y_miss, Y)\nY_miss[rand(1:length(Y_miss), n)] .= missing\n\nmodel = MRVCModel(Y_miss, X, V; se = false)\n@timev fit!(model)","category":"page"},{"location":"examples/#Special-case:-m-2","page":"Examples","title":"Special case: m = 2","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"When there are two variance components, you can accelerate fitting by avoiding large matrix inversion per iteration. To illustrate this, you can first simulate data as done previously but with larger nd and m = 2.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"function simulate(n, d, p, m)\n    X = rand(n, p)\n    B = rand(p, d)\n    V = [zeros(n, n) for _ in 1:m]\n    Σ = [zeros(d, d) for _ in 1:m]\n    Ω = zeros(n * d, n * d)\n    for i in 1:m\n        Vi = randn(n, n)\n        mul!(V[i], transpose(Vi), Vi)\n        Σi = randn(d, d)\n        mul!(Σ[i], transpose(Σi), Σi)\n        kron_axpy!(Σ[i], V[i], Ω) # Ω = Σ[1]⊗V[1] + ... + Σ[m]⊗V[m]\n    end\n    Ωchol = cholesky(Ω)\n    Y = X * B + reshape(Ωchol.L * randn(n * d), n, d)\n    Y, X, V, B, Σ\nend\nY, X, V, B, Σ = simulate(5_000, 4, 10, 2)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Then you can fit data as follows:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = MRTVCModel(Y, X, V)\n@timev fit!(model)\nreduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:2])","category":"page"},{"location":"advanced/#Advanced-details","page":"Advanced details","title":"Advanced details","text":"","category":"section"},{"location":"advanced/#Estimation","page":"Advanced details","title":"Estimation","text":"","category":"section"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"For the MM algorithm, the updates in each iteration are","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\ntextvec boldsymbolB^(t) = (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-(t) (boldsymbolI_d otimes boldsymbolX)^-1 (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-(t) textvec boldsymbolY \nboldsymbolGamma_i^(t + 1) = boldsymbolL_i^-(t)TboldsymbolL_i^(t)TboldsymbolGamma_i^(t)(boldsymbolR^(t)TboldsymbolV_iboldsymbolR^(t))boldsymbolGamma_i^(t)boldsymbolL_i^(t)^12 boldsymbolL_i^-(t)\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"where boldsymbolOmega^(t) = sum_i=1^m boldsymbolGamma_i^(t) otimes boldsymbolV_i, boldsymbolL_i^(t) is the Cholesky factor of boldsymbolM_i^(t) = (boldsymbolI_d otimes boldsymbol1_n)^T (boldsymbol1_d boldsymbol1_d^T otimes boldsymbolV_i) odot boldsymbolOmega^-(t) (boldsymbolI_d otimes boldsymbol1_n), and boldsymbolR^(t) is the n times d matrix such that textvec boldsymbolR^(t) = boldsymbolOmega^-(t) textvec(boldsymbolY - boldsymbolX boldsymbolB^(t)). odot denotes the Hadamard product.","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"For the EM algorithm, the updates in each iteration are","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\ntextvec boldsymbolB^(t) = (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-(t) (boldsymbolI_d otimes boldsymbolX)^-1 (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-(t) textvec boldsymbolY \nboldsymbolGamma_i^(t + 1) = frac1r_i boldsymbolGamma_i^(t) ( boldsymbolR^(t)T boldsymbolV_i boldsymbolR^(t) - boldsymbolM_i^(t) ) boldsymbolGamma_i^(t) + boldsymbolGamma_i^(t)\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"where r_i = textrank(boldsymbolV_i). As seen, the updates for mean effects boldsymbolB are the same for MM and EM algorithms.","category":"page"},{"location":"advanced/#Inference","page":"Advanced details","title":"Inference","text":"","category":"section"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"Standard errors for our estimates are calculated using the Fisher information matrix:","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\ntextE left- fracpartial^2partial(textvec boldsymbolB)^T partial(textvec boldsymbolB) mathcalL right = (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-1 (boldsymbolI_d otimes boldsymbolX) \ntextE left - fracpartial^2partial (textvech boldsymbolGamma_i)^T partial (textvec boldsymbolB) mathcalL right = boldsymbol0 \ntextE left - fracpartial^2partial (textvech boldsymbolGamma_j)^T partial (textvech boldsymbolGamma_i) mathcalL right = frac12 boldsymbolU_i^T (boldsymbolOmega^-1 otimes boldsymbolOmega^-1) boldsymbolU_j\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"where textvech boldsymbolGamma_i creates an fracd(d+1)2 times 1 vector from boldsymbolGamma_i by stacking its lower triangular part and boldsymbolU_i = (boldsymbolI_d otimes boldsymbolK_nd otimes boldsymbolI_n) (boldsymbolI_d^2 otimes textvec boldsymbolV_i) boldsymbolD_d. Here, boldsymbolK_nd is the nd times nd commutation matrix and boldsymbolD_d the d^2 times fracd(d+1)2 duplication matrix.","category":"page"},{"location":"advanced/#Special-case:-missing-response","page":"Advanced details","title":"Special case: missing response","text":"","category":"section"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"In the setting of missing response, the adjusted MM updates in each interation are","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\ntextvec boldsymbolB^(t) = (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-(t) (boldsymbolI_d otimes boldsymbolX)^-1 (boldsymbolI_d otimes boldsymbolX^T) boldsymbolOmega^-(t) textvec boldsymbolZ^(t) \nboldsymbolGamma_i^(t + 1) = boldsymbolL_i^-(t)TboldsymbolL_i^(t)TboldsymbolGamma_i^(t)(boldsymbolR^*(t)TboldsymbolV_iboldsymbolR^*(t) + boldsymbolM_i^*(t))boldsymbolGamma_i^(t)boldsymbolL_i^(t)^12 boldsymbolL_i^-(t)\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"where boldsymbolZ^(t) is the completed response matrix from conditional mean, boldsymbolM_i^*(t) = (boldsymbolI_d otimes boldsymbol1_n)^T (boldsymbol1_d boldsymbol1_d^T otimes boldsymbolV_i) odot (boldsymbolOmega^-(t) boldsymbolP^T boldsymbolC^(t)boldsymbolPboldsymbolOmega^-(t)) (boldsymbolI_d otimes boldsymbol1_n), and boldsymbolR^*(t) is the n times d matrix such that textvec boldsymbolR^*(t) = boldsymbolOmega^-(t) textvec(boldsymbolZ^(t) - boldsymbolX boldsymbolB^(t)). Additionally, boldsymbolP is the nd times nd permutation matrix such that boldsymbolP cdot textvec boldsymbolY = beginbmatrix boldsymboly_textobs  boldsymboly_textmis endbmatrix, where boldsymboly_textobs and boldsymboly_textmis are vectors of observed and missing response values, respectively, in column-major order, and the block matrix boldsymbolC^(t) is boldsymbol0 except for a lower-right block consisting of conditional variance. As seen, the MM updates are of similar form to the non-missing response case.","category":"page"},{"location":"advanced/#Special-case:-m-2","page":"Advanced details","title":"Special case: m = 2","text":"","category":"section"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"When there are m = 2 variance components such that boldsymbolOmega = boldsymbolGamma_1 otimes boldsymbolV_1 + boldsymbolGamma_2 otimes boldsymbolV_2, repeated inversion of the nd times nd matrix boldsymbolOmega per iteration can be avoided and reduced to one d times d generalized eigen-decomposition per iteration. Without loss of generality, if we assume boldsymbolV_2 to be positive definite, the generalized eigen-decomposition of the matrix pair (boldsymbolV_1 boldsymbolV_2) yields generalized eigenvalues boldsymbold = (d_1 dots d_n)^T and generalized eigenvectors boldsymbolU such that boldsymbolU^T boldsymbolV_1 boldsymbolU = boldsymbolD = textdiag(boldsymbold) and boldsymbolU^T boldsymbolV_2 boldsymbolU = boldsymbolI_n. Similarly, if we let the generalized eigen-decomposition of (boldsymbolGamma_1^(t) boldsymbolGamma_2^(t)) be (boldsymbolLambda^(t) boldsymbolPhi^(t)) such that boldsymbolPhi^(t)T boldsymbolGamma_1^(t) boldsymbolPhi^(t) = boldsymbolLambda^(t) = textdiag(boldsymbollambda^(t)) and boldsymbolPhi^(t)T boldsymbolGamma_2^(t) boldsymbolPhi^(t) = boldsymbolI_d, then the MM updates in each iteration become","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\ntextvec boldsymbolB^(t) = (boldsymbolPhi^(t)Totimes tildeboldsymbolX)^T (boldsymbolLambda^(t) otimes boldsymbolD + boldsymbolI_d otimes boldsymbolI_n)^-1 (boldsymbolPhi^(t)Totimes tildeboldsymbolX)^-1 \nquad cdot (boldsymbolPhi^(t)Totimes tildeboldsymbolX)^T (boldsymbolLambda^(t) otimes boldsymbolD + boldsymbolI_d otimes boldsymbolI_n)^-1 textvec(tildeboldsymbolY boldsymbolPhi^(t)) \nboldsymbolGamma_i^(t + 1) = boldsymbolL_i^-(t)TboldsymbolL_i^(t)TboldsymbolN_i^(t)TboldsymbolN_i^(t)boldsymbolL_i^(t)^12 boldsymbolL_i^-(t)\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"where tildeboldsymbolX = boldsymbolU^T boldsymbolX, tildeboldsymbolY = boldsymbolU^T boldsymbolY, boldsymbolL_1^(t) is the Cholesky factor of boldsymbolM_1^(t) = boldsymbolPhi^(t)textdiag(texttr(boldsymbolD(lambda_k^(t)boldsymbolD + boldsymbolI_n)^-1) k = 1dots d)boldsymbolPhi^(t)T, boldsymbolL_2^(t) is the Cholesky factor of boldsymbolM_2^(t) = boldsymbolPhi^(t)textdiag(texttr((lambda_k^(t)boldsymbolD + boldsymbolI_n)^-1) k = 1dots d)boldsymbolPhi^(t)T, boldsymbolN_1^(t) = boldsymbolD^12(tildeboldsymbolY - tildeboldsymbolXboldsymbolB)boldsymbolPhi^(t)oslash(boldsymboldboldsymbollambda^(t)T + boldsymbol1_nboldsymbol1_d^T)  boldsymbolLambda^(t)boldsymbolPhi^-(t), and boldsymbolN_2^(t) = (tildeboldsymbolY - tildeboldsymbolXboldsymbolB)boldsymbolPhi^(t)oslash(boldsymboldboldsymbollambda^(t)T + boldsymbol1_nboldsymbol1_d^T)  boldsymbolPhi^-(t). oslash denotes the Hadamard quotient.","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"For the sake of completeness, we note that the EM updates become","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"boldsymbolGamma_i^(t + 1) = frac1r_i ( boldsymbolN_i^(t)T boldsymbolN_i^(t) - boldsymbolGamma_i^(t) boldsymbolL_i^(t)boldsymbolL_i^(t)T boldsymbolGamma_i^(t) ) + boldsymbolGamma_i^(t)","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"Finally, in this setting, the Fisher information matrix is equivalent to","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\ntextE left- fracpartial^2partial(textvec boldsymbolB)^T partial(textvec boldsymbolB) mathcalL right = (boldsymbolPhi^Totimes tildeboldsymbolX)^T (boldsymbolLambda otimes boldsymbolD + boldsymbolI_d otimes boldsymbolI_n)^-1 (boldsymbolPhi^Totimes tildeboldsymbolX) \ntextE left - fracpartial^2partial (textvech  boldsymbolGamma_i)^T partial (textvec boldsymbolB) mathcalL right = boldsymbol0 \ntextE left - fracpartial^2partial (textvech boldsymbolGamma_j)^T partial (textvech boldsymbolGamma_i) mathcalL right = frac12 boldsymbolD_d^T(boldsymbolPhiotimes boldsymbolPhi) textdiag(textvec boldsymbolW_ij) (boldsymbolPhi otimes boldsymbolPhi)^TboldsymbolD_d\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"where boldsymbolW_ij is the d times d matrix that has entries","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"beginaligned\n(boldsymbolW_11)_kl = texttr(boldsymbolD^2(lambda_k boldsymbolD + boldsymbolI_n)^-1(lambda_l boldsymbolD + boldsymbolI_n)^-1) \n(boldsymbolW_12)_kl = texttr(boldsymbolD(lambda_k boldsymbolD + boldsymbolI_n)^-1(lambda_l boldsymbolD + boldsymbolI_n)^-1) \n(boldsymbolW_22)_kl = texttr((lambda_k boldsymbolD + boldsymbolI_n)^-1(lambda_l boldsymbolD + boldsymbolI_n)^-1)\nendaligned","category":"page"},{"location":"advanced/","page":"Advanced details","title":"Advanced details","text":"for 1 leq k l leq d.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MultiResponseVarianceComponentModels","category":"page"},{"location":"#MRVCModels","page":"Home","title":"MRVCModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"MRVCModels.jl is a package for fitting and testing multivariate response variance components linear mixed models of form","category":"page"},{"location":"","page":"Home","title":"Home","text":"textvec boldsymbolY sim mathcalN(textvec(boldsymbolX boldsymbolB) sum_i=1^m boldsymbolGamma_i otimes boldsymbolV_i)","category":"page"},{"location":"","page":"Home","title":"Home","text":"where boldsymbolY and boldsymbolX are n times d response and  n times p predictor matrices, respectively, and boldsymbolV_1 ldots boldsymbolV_m are m known n times n positive semidefinite matrices. textvec boldsymbolY creates an nd times 1 vector from boldsymbolY by stacking its columns and otimes denotes the Kronecker product. The parameters of the model include p times d mean effects boldsymbolB and d times d variance components (boldsymbolGamma_1 dots boldsymbolGamma_m), which MRVCModels.jl estimates through either minorization-maximization (MM) or expectation–maximization (EM) algorithms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"info: Info\nMRVCModels.jl can also handle data with missing response, which destroys the symmetry of the log-likelihood and complicates maximum likelihood estimation. MM algorithm easily adapts to this challenge.","category":"page"},{"location":"","page":"Home","title":"Home","text":"warning: Warning\nMRVCModels.jl is not suitable for biobank-scale data, except in the setting of m = 2. For m  2, we recommend using this package for datasets of size up to n cdot d approx 60000. Further note that large m can affect memory needed when calculating standard errors, since it will require m(nd)^2 storage space.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To use MRVCModels.jl, type:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url = \"https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl.git\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"This documentation was built using Documenter.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Dates # hide\nprintln(\"Documentation built $(Dates.now()) with Julia $(VERSION)\") # hide","category":"page"}]
}