# Variational Inference for Bayesian SVD model

using LinearAlgebra
using Distributions
using ProgressMeter
using Plots

mutable struct SVDParams
    Ubar :: Matrix{Complex{Float64}}
    Vbar :: Matrix{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    Σbar_V :: Matrix{Complex{Float64}}
    s² :: Float64
end

struct SVDHyperParams
    σ²_V :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function init_svdparams(D :: Int64, T :: Int64, K :: Int64)
    Ubar = vcat(I, zeros(Complex{Float64}, D - K, K))
    Vbar = vcat(I, zeros(Complex{Float64}, T - K, K))
    Σbar_U = diagm(ones(Complex{Float64}, K))
    Σbar_V = diagm(ones(Complex{Float64}, K))
    s² = 1.0 + 0.0im
    #return SVDParams(Ubar, Vbar, Σbar_U, Σbar_V, s²)
    return SVDParams(U_K[:, 1:K], (V_K * diagm(L_K))[:, 1:K], Σbar_U, Σbar_V, s²)
end

function loglik(X :: Matrix{Complex{Float64}}, sp :: SVDParams, hp :: SVDHyperParams)
    M = sp.Ubar * sp.Vbar'
    return -(hp.D * hp.T) * log(π) - hp.T * log(sp.s²) - tr((X - M)' * (X - M)) / sp.s²
end

function freeenergy(X :: Matrix{Complex{Float64}}, sp :: SVDParams, hp :: SVDHyperParams)
    # compute variationa free energy
    D, T, K = hp.D, hp.T, hp.K

    fenergy = - K * log(real(det(sp.Σbar_U))) - K * D * log(D) -
              K * log(real(det(sp.Σbar_V))) + K * D * log(hp.σ²_V) -
              K * (D + T) + D * tr(hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) +
              hp.σ²_V ^ (-1) * tr(hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) + D * T * log(sp.s²) +
              (tr(X' * X) - 2 * tr(real(X' * sp.Ubar * sp.Vbar')) +
               tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))) / sp.s²
    return real(fenergy)
end

function update_Ubar!(X :: Matrix{Complex{Float64}}, sp :: SVDParams)
    sp.Ubar = X * sp.Vbar * sp.Σbar_U / sp.s²
end

function update_Vbar!(X :: Matrix{Complex{Float64}}, sp :: SVDParams)
    sp.Vbar = X' * sp.Ubar * sp.Σbar_V / sp.s²
end

function update_Σbar_U!(sp :: SVDParams, hp :: SVDHyperParams)
    sp.Σbar_U = inv(hp.D * I + (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) / sp.s²)
end

function update_Σbar_V!(sp :: SVDParams, hp :: SVDHyperParams)
    sp.Σbar_V = inv(hp.σ²_V ^ (-1) * I + (hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) / sp.s²)
end

function update_s²!(X :: Matrix{Complex{Float64}}, sp :: SVDParams, hp :: SVDHyperParams)
    numer = tr(X' * X) - 2 * tr(real(X' * sp.Ubar * sp.Vbar')) +
            tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))
    sp.s² = real(numer / (hp.D * hp.T))
end

function bayesiansvd(X :: Matrix{Complex{Float64}}, K :: Int64, n_iter :: Int64;
                     σ²_V :: Float64 = 1e5)
    # X: data matrix (D×T Complex Matrix)
    # K: truncation rank (integer)
    # n_iter: the number of iterations of variational inference (integer)

    D, T = size(X)
    sp = init_svdparams(D, T, K)
    hp = SVDHyperParams(σ²_V, D, T, K)

    logliks = Vector{Float64}(undef, n_iter + 1)
    logliks[1] = loglik(X, sp, hp)
    freeenergies = Vector{Float64}(undef, n_iter + 1)
    freeenergies[1] = freeenergy(X, sp, hp)

    progress = Progress(n_iter)
    for i in 1:n_iter
        update_Ubar!(X, sp)
        update_Σbar_U!(sp, hp)
        update_Vbar!(X, sp)
        update_Σbar_V!(sp, hp)
        update_s²!(X, sp, hp)

        freeenergies[i + 1] = freeenergy(X, sp, hp)
        logliks[i + 1] = loglik(X, sp, hp)
        next!(progress)
    end
    return sp, hp, freeenergies, logliks
end

include("./ComplexNormal.jl")

D = 5
T = 10
K = 3
X = [rand(ComplexNormal(0im, 1)) for i in 1:D, j in 1:T]
U_K, L_K, V_K = svd(X)

sp, hp, freeenergies, logliks = bayesiansvd(X, K, 100)

plot(logliks)
plot(freeenergies)

p1 = plot(real.(U_K[:, 1:K]), lw = 2)
p2 = plot(real.(sp.Ubar), lw = 2)
plot(p1, p2)
heatmap(sp.Ubar * sp.Vbar')
