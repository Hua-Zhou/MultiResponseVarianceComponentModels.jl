"""
    fisher_B!(model::MRVCModel)

Compute the sampling variance-covariance `model.Bcov` of regression coefficients `model.B`,
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_B!(
    model::MRVCModel{T}
) where {T}
    Ωinv = model.storage_nd_nd
    G   = model.storage_pd_pd
    # Gram matrix G = (Id⊗X')Ωinv(Id⊗X) = (X'ΩinvᵢⱼX)ᵢⱼ, 1 ≤ i,j ≤ d
    n, d, p = size(model.Y, 1), size(model.Y, 2), size(model.X, 2)
    for j in 1:d
        Ωcidx = ((j - 1) * n + 1):(j * n)
        Gcidx = ((j - 1) * p + 1):(j * p)
        for i in 1:j
            Ωridx = ((i - 1) * n + 1):(i * n)
            Gridx = ((i - 1) * p + 1):(i * p)
            Ωinvᵢⱼ = view(Ωinv, Ωridx, Ωcidx)
            Gᵢⱼ   = view(G, Gridx, Gcidx)
            mul!(model.storage_n_p, Ωinvᵢⱼ, model.X)
            mul!(Gᵢⱼ, transpose(model.X), model.storage_n_p)
        end
    end
    copytri!(G, 'U')
    return copyto!(model.Bcov, pinv(G))
end

"""
    fisher_Σ!(model::MRVCModel)

Compute the sampling variance-covariance `model.Σcov` of variance component estimates `model.Σ`, 
assuming inverse of covariance matrix `model.Ω` is available at `model.storage_nd_nd`.
"""
function fisher_Σ!(
    model::MRVCModel{T}
) where {T}
    Ωinv = model.storage_nd_nd
    n, d, m = size(model.Y, 1), size(model.Y, 2), length(model.VarComp)
    nd = n * d
    np = m * abs2(d)
    for k in 1:m
        for j in 1:d
            storage_nd_nd_k = model.VarComp[k].storage_nd_nd
            mul!(view(storage_nd_nd_k, :, ((j - 1) * n + 1):(j * n)),
                view(Ωinv, :, ((j - 1) * n + 1):(j * n)),
                model.VarComp[k].V)
        end
    end
    # E[-∂logl/∂vechΣ[j]'∂vechΣ[i] = 1/2 Dd'W[j]'(Ωinv⊗Ωinv)W[i]Dd,
    # where W[i] = Id⊗[(Id⊗V[i])Kdn]^(n) and U[i] = W[i]⋅Dd in manuscript
    Fisher = zeros(T, np, np) # this is a huge matrix md² by md²
    @inbounds for i in 1:np
        # compute 1/2 W[j]'(Ωinv⊗Ωinv)W[i]
        # keep track of indices for each column of W[i]
        k1     = div(i - 1, abs2(d)) + 1 # 1 ≤ k1 ≤ m to choose V[k]
        d2idx1 = mod1(i, abs2(d)) # 1 ≤ d2idx ≤ d² to choose column of W[i]
        ddidx1 = div(d2idx1 - 1, d) # 0 ≤ ddidx ≤ d - 1 to choose columns of Ωinv
        dridx1 = mod1(d2idx1, d) # 1 ≤ dridx ≤ d to choose columns of Ωinv
        for j in i:np
            k2     = div(j - 1, abs2(d)) + 1
            d2idx2 = mod1(j, abs2(d))
            ddidx2 = div(d2idx2 - 1, d)
            dridx2 = mod1(d2idx2, d)
            for (col, row) in enumerate((n * ddidx2 + 1):(n * ddidx2 + n))
                storage_nd_nd_k1 = model.VarComp[k1].storage_nd_nd
                storage_nd_nd_k2 = model.VarComp[k2].storage_nd_nd
                Fisher[i, j] += dot(
                    view(storage_nd_nd_k1,
                        row, (ddidx1 * n + 1):(ddidx1 * n + n)),
                    view(storage_nd_nd_k2,
                        (n * (dridx1 - 1) + 1):(n * dridx1),
                        dridx2 * n - n + col),
                )
            end
            Fisher[i, j] /= 2
        end
    end
    copytri!(Fisher, 'U')
    # # compute 1/2 Dd'W[j]'(Ωinv⊗Ωinv)W[i]Dd
    # vechFisher = zeros(T, m * ◺(d), m * ◺(d))
    # D = duplication(d)
    # for i in 1:m
    #     idx1 = ◺(d) * (i - 1) + 1
    #     idx2 = ◺(d) * i
    #     idx5, idx6 = d^2 * (i - 1) + 1, d^2 * i
    #     for j in i:m
    #         idx3 = ◺(d) * (j - 1) + 1
    #         idx4 = ◺(d) * j
    #         idx7, idx8 = d^2 * (j - 1) + 1, d^2 * j
    #         vechFisher[idx1:idx2, idx3:idx4] = D' * Fisher[idx5:idx6, idx7:idx8] * D
    #     end
    # end
    # copytri!(vechFisher, 'U')
    # copyto!(model.Σcov, pinv(vechFisher))
    return Fisher
end

function update_fisher_info!(model::MRVCModel{T}) where {T}
    Ωinv = model.storage_nd_nd
    ℐ = model.storage_md²_md²
    d = size(model.Y, 2)
    update_Ωinv_V!(Ωinv, model.VarComp)
    update_fisher_info_ltri!(abs2(d), ℐ, model.VarComp)
    return copytri!(ℐ, 'L')
end

@generated function update_Ωinv_V!(
    Ωinv::Matrix{T},
    VarComp::VC,
) where {T,VC<:Tuple}
    K = length(VC.parameters)
    op = [:(_update_Ωinv_V!(Ωinv, VarComp[$i])) for i in 1:K]
    quote
        $(Expr(:meta, :propagate_inbounds))
        begin
            $(op...)
        end
    end
end

function _update_Ωinv_V!(
    Ωinv::Matrix{T},
    VC::VarCompStructure{T},
) where {T}
    n = size(VC.V, 1)
    d = size(VC.Σ, 1)
    fill!(VC.storage_nd_nd, zero(T))
    for j in 1:d, i in j:d
        jstart = (j - 1) * n + 1
        jstop = j * n
        istart = (i - 1) * n + 1
        istop = i * n
        Ωinv_ij = view(Ωinv, istart:istop, jstart:jstop)
        out = view(VC.storage_nd_nd, istart:istop, jstart:jstop)
        BLAS.symm!('R', 'L', one(T), VC.V, Ωinv_ij, zero(T), out)
    end
end

@generated function update_fisher_info_ltri!(
    d²::Int, ℐ::Matrix{T}, VarComp::VC
) where {T,VC<:Tuple}
    K = length(VC.parameters)
    op = Expr[]
    for j in 1:K
        for i in j:K
            push!(
                op,
                :(_update_fisher_block!(
                    view(
                        ℐ,
                        (($i - 1) * d² + 1):($i * d²),
                        (($j - 1) * d² + 1):($j * d²),
                    ),
                    VarComp[$i],
                    VarComp[$j],
                )),
            )
        end
    end
    quote
        $(Expr(:meta, :propagate_inbounds))
        begin
            $(op...)
        end
        return nothing
    end
end

function _update_fisher_block!(
    ℐpq::AbstractMatrix{T},
    VCp::VarCompStructure{T},
    VCq::VarCompStructure{T},
) where {T}
    n = size(VCp.storage_nn, 1)
    d = size(VCp, 1)
    Ωinv_Vp = VCp.storage_nd_nd
    Ωinv_Vq = VCq.storage_nd_nd
    @inbounds for l in 1:d
        for k in l:d
            lstart = (l - 1) * n + 1
            lstop = l * n
            kstart = (k - 1) * n + 1
            kstop = k * n
            Ωinv_kl = view(Ωinv_Vp, kstart:kstop, lstart:lstop)
            for j in 1:d
                for i in j:d
                    jstart = (j - 1) * n + 1
                    jstop = j * n
                    istart = (i - 1) * n + 1
                    istop = i * n
                    Ωinv_ij = view(Ωinv_Vq, istart:istop, jstart:jstop)
                    cidx = (l - 1) * d + j
                    ridx = (k - 1) * d + i
                    ℐpq[ridx, cidx] = dot(transpose(Ωinv_ij), Ωinv_kl)
                end
            end
            copytri!(
                view(ℐpq, ((k - 1) * d + 1):(k * d), ((l - 1) * d + 1):(l * d)),
                'L',
            )
        end
    end
    ℐpq ./= 2
    return copytri!(ℐpq, 'L')
end