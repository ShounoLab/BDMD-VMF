using Distributions
using ProgressMeter

mutable struct DMDParams
    λ :: Vector{Complex{Float64}}
    W :: Matrix{Complex{Float64}}
    σ² :: Float64
end

struct DMDHyperParams
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


function loglik(X :: Matrix{Complex{Float64}}, dp :: DMDParams,
                hp :: DMDHyperParams)
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

function logprior(dp :: DMDParams, hp :: DMDHyperParams)
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

function metropolis!(X :: Matrix{Complex{Float64}}, dp :: DMDParams,
                     dp_cand :: DMDParams, hp :: DMDHyperParams)
    logp_orig = loglik(X, dp, hp) + logprior(dp, hp)
    logp_cand = loglik(X, dp_cand, hp) + logprior(dp_cand, hp)

    logr = logp_cand - logp_orig
    if logr > 0
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    elseif logr > log(rand())
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    end
end

function metropolis_λ!(X :: Matrix{Complex{Float64}}, dp :: DMDParams,
                       hp :: DMDHyperParams)
    σₚᵣₒₚ = 1e-2
    for k in 1:hp.K
        dp_cand = deepcopy(dp)
        dp_cand.λ[k] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
        metropolis!(X, dp, dp_cand, hp)
    end
end

function metropolis_W!(X :: Matrix{Complex{Float64}}, dp :: DMDParams,
                       hp :: DMDHyperParams)
    σₚᵣₒₚ = 0.1
    @views for k in 1:hp.K
        for l in 1:hp.K
            dp_cand = deepcopy(dp)
            dp_cand.W[k, l] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
            metropolis!(X, dp, dp_cand, hp)
        end
    end
end

function metropolis_σ²!(X :: Matrix{Complex{Float64}}, dp :: DMDParams,
                        hp :: DMDHyperParams)
    σₚᵣₒₚ = 1e-2
    dp_cand = deepcopy(dp)
    dp_cand.σ² += rand(Normal(0.0, σₚᵣₒₚ))
    if dp_cand.σ² > 0
        metropolis!(X, dp, dp_cand, hp)
    end
end

function init_dmdparams(hp :: DMDHyperParams)
    λ = ones(Complex{Float64}, hp.K)
    W = zeros(Complex{Float64}, (hp.K, hp.K))
    σ² = 1.0
    return DMDParams(λ, W, σ²)
end

function run_sampling(X :: Matrix{Complex{Float64}}, hp :: DMDHyperParams,
                      n_iter :: Int64)
    dp = init_dmdparams(hp)

    dp_ary = Vector{DMDParams}(undef, n_iter)
    logliks = Vector{Float64}(undef, n_iter)

    progress = Progress(n_iter)
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
