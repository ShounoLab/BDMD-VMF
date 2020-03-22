using Distributions
using ProgressMeter
using SparseArrays
using KernelDensity
using PDMats

mutable struct BDMDParams
    λ :: Vector{Complex{Float64}}
    W :: Matrix{Complex{Float64}}
    σ² :: Float64
end

struct BDMDHyperParams
    vecUbar :: Vector{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    Γbar_U :: Matrix{Complex{Float64}}
    Σbar_U_inv :: Matrix{Complex{Float64}}
    Γbar_U_inv :: Matrix{Complex{Float64}}
    σ²_λ :: Float64
    σ²_w :: Float64
    α :: Float64
    β :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function BDMDHyperParams(sp :: SVDParams, shp :: SVDHyperParams;
                         σ²_λ :: Float64 = 1e5, σ²_w :: Float64 = 1e5,
                         α :: Float64 = 0.01, β :: Float64 = 0.01)
    # outer constructor
    C = commutation_matrix(shp.D, shp.K)
    Σbar_U = zeros(Complex{Float64}, (shp.D * shp.K, shp.D * shp.K))
    Σbar_U_inv = zeros(Complex{Float64}, (shp.D * shp.K, shp.D * shp.K))

    map(d -> Σbar_U[((d - 1) * K + 1):(d * K), ((d - 1) * K + 1):(d * K)] =
             deepcopy(sp.Σbar_U[d, :, :]), 1:shp.D)
    map(d -> Σbar_U_inv[((d - 1) * K + 1):(d * K), ((d - 1) * K + 1):(d * K)] =
        deepcopy(sp.Σbar_U_inv[d, :, :]), 1:shp.D)

    Γᵤ⁻¹ = C' * transpose(Σbar_U_inv) * C
    Γᵤ = inv(Γᵤ⁻¹)
    return BDMDHyperParams(vec(sp.Ubar), Σbar_U, Γᵤ, Σbar_U_inv, Γᵤ⁻¹,
                           σ²_λ, σ²_w, α, β,
                           shp.D, shp.T, shp.K)
end

function commutation_matrix(M :: Int64, N :: Int64)
    # get commutation matrix of vectrized matrix
 

    # see Def.2.1 in Magnus and Neudecker (1979) for details

    A = reshape(1:(M * N), (M, N))'
    A = vec(A)

    return sparse(diagm(ones(Int64, M * N))[A, :])
end

function sparsechol(A :: SparseMatrixCSC{Complex{Float64}},
                    shift :: Float64 = 1e-3)
    return cholesky(Hermitian(A), shift = shift)
end

function logdetΣ(Σ :: SparseMatrixCSC{Complex{Float64}})
    try
        return logdet(sparsechol(Σ))
    catch
        if isa(e, PosDefException)
            return log(det(lu(sparse(A))))
        end
    end
end

function fact_inv(A :: SparseMatrixCSC{Complex{Float64}})
    try
        L = Matrix(sparse(sparsechol(A).L))
        invL = inv(L)
        return invL' * invL
    catch e
        if isa(e, PosDefException)
            return inv(lu(A))
        end
    end
end

function fact_inv_logdet(A :: SparseMatrixCSC{Complex{Float64}})
    #try
    #    println("cholesky")
    #    F = sparsechol(A)
    #    L = Matrix(sparse(F.PtL))
    #    invL = inv(L)
    #    return sparse(invL' * invL), -logdet(F)
    #catch e
    #    if isa(e, PosDefException)
    #        F = lu(A)
    #        println("LU")
    #        return sparse(inv(F)), -log(det(F))
    #    end
    #end
    F = lu(A)
    return sparse(inv(F)), -sum(log.(diag(F.U ./ F.Rs)))
end

function loglik(X :: Matrix{Union{Missing, Complex{Float64}}},
                dp :: BDMDParams, hp :: BDMDHyperParams)
    vecX = vec(X)
    missvec = .!ismissing.(vecX)
    vecX_com = vecX[missvec]
    Nmiss = sum(missvec)

    I_D = spdiagm(0 => ones(hp.D))
    G = zeros(Complex{Float64}, hp.K, hp.T)
    map(t -> G[:, t] = dp.W * (dp.λ .^ (t - 1)), 1:hp.T)
    GtId = kron(sparse(transpose(G)), I_D)
    GtId_com = GtId[missvec, :]
    GtId_com2 = GtId_com' * GtId_com

    Γᵤ⁻¹ = sparse(hp.Γbar_U_inv)

    Σᵤ⁻¹ = GtId_com2 / dp.σ² + Γᵤ⁻¹
    Σᵤ, logdetΣᵤ = fact_inv_logdet(Σᵤ⁻¹)

    #Σₓ⁻¹_com = I ./ dp.σ² - GtId_com * Σᵤ * GtId_com

    Σₓ⁻¹_com = - GtId_com * Σᵤ * GtId_com' / (dp.σ² ^ 2)
    map(i -> Σₓ⁻¹_com[i, i] += inv(dp.σ²), 1:Nmiss)

    # inversion of Σₓ⁻¹ : Woodbury formula
    # Σₓ_com = I * dp.σ² + GtId_com * inv(Σᵤ⁻¹ - GtId_com2 / dp.σ²) * GtId_com'
    Σ₁_com, logdetΣ₁_com = fact_inv_logdet(Σᵤ⁻¹ - GtId_com2 / dp.σ²)
    Σₓ_com = GtId_com * Σ₁_com * GtId_com'
    map(i -> Σₓ_com[i, i] += dp.σ², 1:Nmiss)

    vecXbar_com = Σₓ_com * GtId_com * Σᵤ * Γᵤ⁻¹ * hp.vecUbar / dp.σ²

    # determinant lemma
    logdetΣₓ_com = Nmiss * log(dp.σ²) - logdetΣᵤ + logdetΣ₁_com

    logL = -real(Nmiss * log(π * dp.σ²) + logdetΣₓ_com +
                (vecX_com - vecXbar_com)' * Σₓ⁻¹_com * (vecX_com - vecXbar_com))
    return logL
end

function logprior(dp :: BDMDParams, hp :: BDMDHyperParams)
    logp = 0.0
    for k in 1:hp.K
        logp += loglikelihood(ComplexNormal(0.0im, √(hp.σ²_λ)), dp.λ[k])
        logp += loglikelihood(MvComplexNormal(zeros(Complex{Float64}, hp.K),
                                              √(hp.σ²_w)),
                              dp.W[:, k])
    end
    logp += logpdf(InverseGamma(hp.α, hp.β), dp.σ²)
    return logp
end

function metropolis!(X :: Matrix{Union{Missing, Complex{Float64}}},
                     dp :: BDMDParams, dp_cand :: BDMDParams, hp :: BDMDHyperParams)
    logp_orig = loglik(X, dp, hp) + logprior(dp, hp)
    logp_cand = loglik(X, dp_cand, hp) + logprior(dp_cand, hp)

    logr = logp_cand - logp_orig
    if logr > 0
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    elseif logr > log(rand())
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    end
end

function metropolis_λ!(X :: Matrix{Union{Missing, Complex{Float64}}},
                       dp :: BDMDParams, hp :: BDMDHyperParams)
    σₚᵣₒₚ = 1e-2
    for k in 1:hp.K
        dp_cand = deepcopy(dp)
        dp_cand.λ[k] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
        metropolis!(X, dp, dp_cand, hp)
    end
end

function metropolis_W!(X :: Matrix{Union{Missing, Complex{Float64}}},
                       dp :: BDMDParams, hp :: BDMDHyperParams)
    σₚᵣₒₚ = 1e-1
    @views for k in 1:hp.K
        for l in 1:hp.K
            dp_cand = deepcopy(dp)
            dp_cand.W[k, l] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
            metropolis!(X, dp, dp_cand, hp)
        end
    end
end

function metropolis_σ²!(X :: Matrix{Union{Missing, Complex{Float64}}},
                        dp :: BDMDParams, hp :: BDMDHyperParams)
    σₚᵣₒₚ = 1e-2
    dp_cand = deepcopy(dp)
    dp_cand.σ² += rand(Normal(0.0, σₚᵣₒₚ))
    if dp_cand.σ² > 0
        metropolis!(X, dp, dp_cand, hp)
    end
end

function init_dmdparams(hp :: BDMDHyperParams)
    λ = ones(Complex{Float64}, hp.K) .* 0.5
    W = zeros(Complex{Float64}, (hp.K, hp.K))
    σ² = 1.0
    return BDMDParams(λ, W, σ²)
end

function run_sampling(X :: Matrix{Union{Missing, Complex{Float64}}},
                      hp :: BDMDHyperParams, n_iter :: Int64)
    dp = init_dmdparams(hp)

    dp_ary = Vector{BDMDParams}(undef, n_iter)
    logliks = Vector{Float64}(undef, n_iter)

    progress = Progress(n_iter)

    dp.W = deepcopy(Matrix(transpose(transpose(naive_dp.Φ) .* naive_dp.b)))
    dp.λ = deepcopy(naive_dp.λ)
    for i in 1:n_iter
        metropolis_W!(X, dp, hp)
        metropolis_λ!(X, dp, hp)
        metropolis_σ²!(X, dp, hp)

        dp_ary[i] = deepcopy(dp)
        logliks[i] = loglik(X, dp, hp)
        next!(progress)
    end

    return dp_ary, logliks
end

#function reconstruct(dp_ary :: Vector{BDMDParams},
#                     hp :: BDMDHyperParams, sp :: SVDParams,
#                     N :: Int64, burnin :: Int64)
#    X_preds = Array{ComplexF64, 3}(undef, (hp.D, hp.T, N))
#    progress = Progress(N)
#    for n in 1:N
#        vecUbar = rand(MvComplexNormal(hp.vecUbar, hp.Γbar_U, check_posdef = false, check_hermitian = false))
#        Ubar = reshape(vecUbar, (hp.D, hp.K))
#
#        ind = rand((burnin + 1):length(dp_ary))
#        λ, W, σ² = dp_ary[ind].λ, dp_ary[ind].W, dp_ary[ind].σ²
#
#        I_D = spdiagm(0 => ones(hp.D))
#        G = zeros(Complex{Float64}, hp.K, hp.T)
#        map(t -> G[:, t] = W * (λ .^ (t - 1)), 1:hp.T)
#
#        X_preds[:, :, n] .= reshape(rand(MvComplexNormal(vec(Ubar * G),
#                                                         spdiagm(0 => fill(σ², hp.D * hp.T)),
#                                                         check_posdef = false,
#                                                         check_hermitian = false)),
#                                    (hp.D, hp.T))
#        next!(progress)
#    end
#    return X_preds
#end


function reconstruct(dp_ary :: Vector{BDMDParams},
                     hp :: BDMDHyperParams, sp :: SVDParams,
                     N :: Int64, burnin :: Int64)
    X_preds = Array{ComplexF64, 3}(undef, (hp.D, hp.T, N))
    progress = Progress(N)
    for n in 1:N
        ind = rand((burnin + 1):length(dp_ary))
        λ, W, σ² = dp_ary[ind].λ, dp_ary[ind].W, dp_ary[ind].σ²

        I_D = spdiagm(0 => ones(hp.D))
        G = zeros(Complex{Float64}, hp.K, hp.T)
        map(t -> G[:, t] = W * (λ .^ (t - 1)), 1:hp.T)
        GtId = kron(sparse(transpose(G)), I_D)
        GtId2 = GtId' * GtId

        Γᵤ⁻¹ = sparse(hp.Γbar_U_inv)

        Σᵤ⁻¹ = GtId2 / σ² + Γᵤ⁻¹
        Σᵤ, logdetΣᵤ = fact_inv_logdet(Σᵤ⁻¹)
        Σₓ⁻¹ = - GtId * Σᵤ * GtId' / (σ² ^ 2)
        map(i -> Σₓ⁻¹[i, i] += inv(σ²), 1:(hp.D * hp.T))

        # inversion of Σₓ⁻¹ : Woodbury formula
        Σ₁, logdetΣ₁ = fact_inv_logdet(Σᵤ⁻¹ - GtId2 / σ²)
        Σₓ = GtId * Σ₁ * GtId'
        map(i -> Σₓ[i, i] += σ², 1:(hp.D * hp.T))

        vecXbar = Σₓ * GtId * Σᵤ * Γᵤ⁻¹ * hp.vecUbar / σ²

        heatmap(real.(reshape(vecXbar, (hp.D, hp.T))))
        X_preds[:, :, n] .= reshape(rand(MvComplexNormal(vecXbar, Σₓ, check_posdef = false,
                                                   check_hermitian = false)),
                              (hp.D, hp.T))
        next!(progress)
    end
    return X_preds
end

function map_bdmd(dp_ary :: Vector{BDMDParams}, hp :: BDMDHyperParams,
                  burnin :: Int64)
    N = length(dp_ary)
    λ = Vector{Complex{Float64}}(undef, hp.K)
    W = Matrix{Complex{Float64}}(undef, hp.K, hp.K)
    for k in 1:hp.K
        λ_ary = [dp_ary[i].λ[k] for i in (burnin + 1):N]
        ker_λ = kde((real.(λ_ary), imag.(λ_ary)))
        maxind_λ = findmax(ker_λ.density)[2]
        λ[k] = ker_λ.x[maxind_λ[1]] + im * ker_λ.y[maxind_λ[2]]
        for l in 1:hp.K
            w_ary = [dp_ary[i].W[k, l] for i in (burnin + 1):N]
            ker_w = kde((real.(w_ary), imag.(w_ary)))
            maxind_w = findmax(ker_w.density)[2]
            W[k, l] = ker_w.x[maxind_w[1]] + im * ker_w.y[maxind_w[2]]
        end
    end
    σ²_ary = [dp_ary[i].σ² for i in (burnin + 1):N]
    ker_σ² = kde(σ²_ary)
    σ² = ker_σ².x[findmax(ker_σ².density)[2]]
    return BDMDParams(λ, W, σ²)
end

function reconstruct_map(dp :: BDMDParams, hp :: BDMDHyperParams)
    I_D = spdiagm(0 => ones(hp.D))
    G = zeros(Complex{Float64}, hp.K, hp.T)
    map(t -> G[:, t] = dp.W * (dp.λ .^ (t - 1)), 1:hp.T)
    GtId = kron(sparse(transpose(G)), I_D)
    GtId2 = GtId' * GtId

    Γᵤ⁻¹ = sparse(hp.Γbar_U_inv)

    Σᵤ⁻¹ = GtId2 / dp.σ² + Γᵤ⁻¹
    Σᵤ, logdetΣᵤ = fact_inv_logdet(Σᵤ⁻¹)
    Σₓ⁻¹ = - GtId * Σᵤ * GtId' / (dp.σ² ^ 2)
    map(i -> Σₓ⁻¹[i, i] += inv(dp.σ²), 1:(hp.D * hp.T))

    # inversion of Σₓ⁻¹ : Woodbury formula
    Σ₁, logdetΣ₁ = fact_inv_logdet(Σᵤ⁻¹ - GtId2 / dp.σ²)
    Σₓ = GtId * Σ₁ * GtId'
    map(i -> Σₓ[i, i] += dp.σ², 1:(hp.D * hp.T))

    vecXbar = Σₓ * GtId * Σᵤ * Γᵤ⁻¹ * hp.vecUbar / dp.σ²
    return Matrix(reshape(vecXbar, hp.D, hp.T))
end

function get_quantiles(X_preds :: Array{ComplexF64}; interval = 0.95)
    α = (1 - interval) / 2
    D, T = size(X_preds)[1], size(X_preds)[2]
    X_quantiles_real = Array{Float64, 3}(undef, (D, T, 2))
    X_quantiles_imag = Array{Float64, 3}(undef, (D, T, 2))
    for d in 1:D
        map(t -> X_quantiles_real[d, t, :] = quantile(real.(X_preds[d, t, :]), [α, 1-α]), 1:T)
        map(t -> X_quantiles_imag[d, t, :] = quantile(imag.(X_preds[d, t, :]), [α, 1-α]), 1:T)
    end
    return X_quantiles_real, X_quantiles_imag
end
