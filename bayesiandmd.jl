using Distributions
using ProgressMeter

mutable struct BDMDParams
    λ :: Vector{Complex{Float64}}
    W :: Matrix{Complex{Float64}}
    σ² :: Float64
end

struct BDMDHyperParams
    Ubar :: Matrix{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    σ²_λ :: Float64
    σ²_w:: Float64
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
    return BDMDHyperParams(sp.Ubar, sp.Σbar_U, σ²_λ, σ²_w, α, β,
                          shp.D, shp.T, shp.K)
end

struct MCMCConfig
    n_iter :: Int64
    burnin :: Int64
    σₚᵣₒₚ_λ :: Float64
    σₚᵣₒₚ_W :: Float64
    σₚᵣₒₚ_σ² :: Float64
    σₚᵣₒₚ_λ_burnin :: Float64
    σₚᵣₒₚ_W_burnin :: Float64
    σₚᵣₒₚ_σ²_burnin :: Float64
end

function MCMCConfig(n_iter :: Int64, burnin :: Int64,
                    σₚᵣₒₚ_λ :: Float64, σₚᵣₒₚ_W :: Float64,
                    σₚᵣₒₚ_σ² :: Float64)
    return MCMCConfig(n_iter :: Int64, burnin :: Int64,
                      σₚᵣₒₚ_λ :: Float64, σₚᵣₒₚ_W :: Float64,
                      σₚᵣₒₚ_σ² :: Float64, σₚᵣₒₚ_λ :: Float64,
                      σₚᵣₒₚ_W :: Float64,σₚᵣₒₚ_σ² :: Float64)
end



function loglik(X :: Matrix{Complex{Float64}}, dp :: BDMDParams,
                hp :: BDMDHyperParams)
    logL = 0
    for t in 1:hp.T
        gₜ = dp.W * (dp.λ .^ t)
        Gₜ = (gₜ * gₜ' / dp.σ² + hp.Σbar_U ^ (-1)) ^ (-1)
        σₜ² = real(dp.σ² * (1 - gₜ' * Gₜ * gₜ) ^ (-1))
        xₜ = σₜ² / dp.σ² * hp.Ubar * hp.Σbar_U ^ (-1) * Gₜ * gₜ
        logL += loglikelihood(MvComplexNormal(xₜ, √(σₜ²)), X[:, t])
    end
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

function metropolis!(X :: Matrix{Complex{Float64}}, dp :: BDMDParams,
                     dp_cand :: BDMDParams, hp :: BDMDHyperParams)
    logp_orig = loglik(X, dp, hp) + logprior(dp, hp)
    logp_cand = loglik(X, dp_cand, hp) + logprior(dp_cand, hp)

    logr = logp_cand - logp_orig
    if sum(norm.(dp_cand.λ) .^ 2 .> 1.0) >= 1
        return nothing
    end
    if logr > 0
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    elseif logr > log(rand())
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    end
end

function metropolis_λ!(X :: Matrix{Complex{Float64}}, dp :: BDMDParams,
                       hp :: BDMDHyperParams, mc :: MCMCConfig, iter :: Int64)
    σₚᵣₒₚ = ifelse(iter > mc.burnin, mc.σₚᵣₒₚ_λ, mc.σₚᵣₒₚ_λ_burnin)
    for k in 1:hp.K
        dp_cand = deepcopy(dp)
        dp_cand.λ[k] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
        metropolis!(X, dp, dp_cand, hp)
    end
end

function metropolis_W!(X :: Matrix{Complex{Float64}}, dp :: BDMDParams,
                       hp :: BDMDHyperParams, mc :: MCMCConfig, iter :: Int64)
    σₚᵣₒₚ = ifelse(iter > mc.burnin, mc.σₚᵣₒₚ_W, mc.σₚᵣₒₚ_W_burnin)
    @views for k in 1:hp.K
        for l in 1:hp.K
            dp_cand = deepcopy(dp)
            dp_cand.W[k, l] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
            metropolis!(X, dp, dp_cand, hp)
        end
    end
end

function metropolis_σ²!(X :: Matrix{Complex{Float64}}, dp :: BDMDParams,
                        hp :: BDMDHyperParams, mc :: MCMCConfig, iter :: Int64)
    σₚᵣₒₚ = ifelse(iter > (mc.burnin - 1000), mc.σₚᵣₒₚ_σ², mc.σₚᵣₒₚ_σ²_burnin)
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

function run_sampling(X :: Matrix{Complex{Float64}}, hp :: BDMDHyperParams,
                      mc :: MCMCConfig)
    dp = init_dmdparams(hp)

    dp_ary = Vector{BDMDParams}(undef, mc.n_iter)
    logliks = Vector{Float64}(undef, mc.n_iter)

    progress = Progress(mc.n_iter)
    for i in 1:mc.n_iter
        metropolis_W!(X, dp, hp, mc, i)
        metropolis_λ!(X, dp, hp, mc, i)
        metropolis_σ²!(X, dp, hp, mc, i)

        dp_ary[i] = deepcopy(dp)
        logliks[i] = loglik(X, dp, hp)
        next!(progress)
    end

    return dp_ary, logliks
end


function mean_bdmd(dp_ary :: Vector{BDMDParams}, hp :: BDMDHyperParams,
                   mc :: MCMCConfig)
    N = length(dp_ary)
    λ = Vector{Complex{Float64}}(undef, hp.K)
    W = Matrix{Complex{Float64}}(undef, hp.K, hp.K)
    for k in 1:hp.K
        λ[k] = mean([dp_ary[i].λ[k] for i in (mc.burnin + 1):N])
        for l in 1:hp.K
            W[k, l] = mean([dp_ary[i].W[k, l] for i in (mc.burnin + 1):N])
        end
    end
    σ² = mean([dp_ary[i].σ² for i in (mc.burnin + 1):N])
    return BDMDParams(λ, W, σ²)
end


function reconstruct_pointest(dp :: BDMDParams, hp :: BDMDHyperParams)
    X_reconst_bdmd = Matrix{ComplexF64}(undef, size(X))
    λ, W, σ² = dp_ary[end].λ, dp_ary[end].W, dp_ary[end].σ²
    for t in 1:hp.T
        gₜ = dp.W * (dp.λ .^ t)
        Gₜ = (gₜ * gₜ' / dp.σ² + hp.Σbar_U ^ (-1)) ^ (-1)
        σₜ² = real(dp.σ² * (1 - gₜ' * Gₜ * gₜ) ^ (-1))
        xₜ = σₜ² / dp.σ² * hp.Ubar * hp.Σbar_U ^ (-1) * Gₜ * gₜ
        X_reconst_bdmd[:, t] .= xₜ
    end

    return X_reconst_bdmd
end


