# Variational Inference for Bayesian PCA model

using LinearAlgebra
using Distributions
using ProgressMeter
using Plots
using MultivariateStats
using RDatasets

mutable struct PCAParams
    Ubar :: Matrix{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    Zbar :: Matrix{Complex{Float64}}
    Σbar_Z :: Matrix{Complex{Float64}}
    μbar :: Vector{Complex{Float64}}
    Σbar_μ :: Matrix{Complex{Float64}}
    s² :: Float64
end

struct PCAHyperParams
    σ²_U :: Float64
    σ²_Z :: Float64
    σ²_μ :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function init_pcaparams(D :: Int64, T :: Int64, K :: Int64)
    Ubar = [rand(ComplexNormal()) for i in 1:D, j in 1:K]
    Zbar = [rand(ComplexNormal()) for i in 1:K, j in 1:T]
    μbar = [rand(ComplexNormal()) for i in 1:D]
    Σbar_U = diagm(ones(Complex{Float64}, K))
    Σbar_Z = diagm(ones(Complex{Float64}, K))
    Σbar_μ = diagm(ones(Complex{Float64}, D))
    s² = 1.0
    return PCAParams(Ubar, Σbar_U, Zbar, Σbar_Z, μbar, Σbar_μ, s²)
end

function loglik(X :: Matrix{Complex{Float64}}, pp :: PCAParams, hp :: PCAHyperParams)
    Xbar = pp.Ubar * pp.Zbar .+ repeat(pp.μbar, outer = (1, hp.T))
    return -(hp.D * hp.T) * log(π) - hp.T * log(pp.s²) - tr((X - Xbar)' * (X - Xbar)) / pp.s²
end

#function freeenergy(X :: Matrix{Complex{Float64}}, pp :: PCAParams, hp :: PCAHyperParams)
#    # compute variationa free energy
#    D, T, K = hp.D, hp.T, hp.K
#
#    fenergy = - K * log(real(det(sp.Σbar_U))) - K * D * log(D) -
#              K * log(real(det(sp.Σbar_V))) + K * D * log(hp.σ²_V) -
#              K * (D + T) + D * tr(hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) +
#              hp.σ²_V ^ (-1) * tr(hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) + D * T * log(sp.s²) +
#              (tr(X' * X) - 2 * tr(real(X' * sp.Ubar * sp.Vbar')) +
#               tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))) / sp.s²
#    return real(fenergy)
#end

function update_Ubar!(X :: Matrix{Complex{Float64}}, pp :: PCAParams, hp :: PCAHyperParams)
    M = repeat(pp.μbar, outer = (1, hp.T))
    pp.Ubar = (X - M) * pp.Zbar' * pp.Σbar_U / pp.s²
end

function update_Zbar!(X :: Matrix{Complex{Float64}}, pp :: PCAParams, hp :: PCAHyperParams)
    M = repeat(pp.μbar, outer = (1, hp.T))
    pp.Zbar = pp.Σbar_Z * pp.Ubar' * (X - M) / pp.s²
end

function update_μbar!(X :: Matrix{Complex{Float64}}, pp :: PCAParams, hp :: PCAHyperParams)
    mμ = zeros(Complex{Float64}, hp.D)
    for t in 1:hp.T
        @views mμ .+= X[:, t] - pp.Ubar * pp.Zbar[:, t]
    end
    pp.μbar = pp.Σbar_μ * mμ / pp.s²
end

function update_Σbar_U!(pp :: PCAParams, hp :: PCAHyperParams)
    pp.Σbar_U = inv(hp.σ²_U ^ (-1) * I + (hp.T * transpose(pp.Σbar_Z) + pp.Zbar * pp.Zbar') / pp.s²)
end

function update_Σbar_Z!(pp :: PCAParams, hp :: PCAHyperParams)
    pp.Σbar_Z = inv(hp.σ²_Z ^ (-1) * I + (hp.D * pp.Σbar_U + pp.Ubar' * pp.Ubar) / pp.s²)
end

function update_Σbar_μ!(pp :: PCAParams, hp :: PCAHyperParams)
    pp.Σbar_μ = inv(hp.σ²_μ ^ (-1) + hp.T / pp.s²) * diagm(ones(hp.D))
end

function update_s²!(X :: Matrix{Complex{Float64}}, pp :: PCAParams, hp :: PCAHyperParams)
    M = repeat(pp.μbar, outer = (1, hp.T))
    numer = tr(X' * X) - 2 * tr(real(X' * (pp.Ubar * pp.Zbar .+ M))) +
            2 * tr(real(M' * pp.Ubar * pp.Zbar)) +
            tr((hp.D * pp.Σbar_U + pp.Ubar' * pp.Ubar) * (hp.T * transpose(pp.Σbar_Z) + pp.Zbar * pp.Zbar')) +
            hp.T * (tr(pp.Σbar_μ) + pp.μbar' * pp.μbar)
    pp.s² = real(numer / (hp.D * hp.T))
end

function bayesianpca(X :: Matrix{Complex{Float64}}, K :: Int64, n_iter :: Int64;
                     σ²_U :: Float64 = 1e5, σ²_Z :: Float64 = 1.0, σ²_μ :: Float64 = 1e5)
    # X: data matrix (D×T Complex Matrix)
    # K: truncation rank (integer)
    # n_iter: the number of iterations of variational inference (integer)

    D, T = size(X)
    pp = init_pcaparams(D, T, K)
    hp = PCAHyperParams(σ²_U, σ²_Z, σ²_μ, D, T, K)

    logliks = Vector{Float64}(undef, n_iter + 1)
    logliks[1] = loglik(X, pp, hp)
    s_ary = Vector{Float64}(undef, n_iter + 1)
    s_ary[1] = pp.s²
    #freeenergies = Vector{Float64}(undef, n_iter + 1)
    #freeenergies[1] = freeenergy(X, sp, hp)

    progress = Progress(n_iter)
    pp.μbar = reshape(mean(X, dims = 2), D)
    for i in 1:n_iter
        update_Σbar_U!(pp, hp)
        update_Σbar_Z!(pp, hp)
        update_Σbar_μ!(pp, hp)
        update_Ubar!(X, pp, hp)
        update_Zbar!(X, pp, hp)
        #update_μbar!(X, pp, hp)
        update_s²!(X, pp, hp)

        #freeenergies[i + 1] = freeenergy(X, sp, hp)
        logliks[i + 1] = loglik(X, pp, hp)
        s_ary[i + 1] = pp.s²
        next!(progress)
    end
    #return pp, hp, freeenergies, logliks
    return pp, hp, logliks, s_ary
end

include("ComplexNormal.jl")
iris = dataset("datasets", "iris")
iris = Matrix(transpose(Matrix{Complex{Float64}}(iris[:, 1:4])))
K = 2

PCAModel = fit(PCA, real.(iris), maxoutdim = K)
PPCAModel = fit(PPCA, real.(iris), maxoutdim = K, method = :em)

pp, hp, logliks, s_ary = bayesianpca(iris, 2, 1000)
M = repeat(pp.μbar, outer = (1, hp.T))

plot(logliks)
plot(s_ary)

p1 = plot(real.(projection(PCAModel)), lw = 2)
p2 = plot(real.(loadings(PPCAModel)), lw = 2)
p3 = plot(real.(pp.Ubar), lw = 2)
plot(p1, p2)
