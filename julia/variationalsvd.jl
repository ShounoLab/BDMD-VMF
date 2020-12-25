# Variational Inference for Bayesian SVD model 
using LinearAlgebra
using Distributions
using ProgressMeter

include("$(@__DIR__)/ComplexNormal.jl")

mutable struct SVDParams{F <: Union{Float64, ComplexF64}}
    Ubar :: Matrix{F}
    Vbar :: Matrix{F}
    Σbar_U :: Matrix{F}
    Σbar_V :: Matrix{F}
    C_V :: Matrix{F}
    s² :: Float64
end

struct SVDHyperParams
    σ²_U :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function init_svdparams(X :: Matrix{F}, D :: Int64, T :: Int64, K :: Int64;
                        svdinit :: Bool = true) where {F <: Union{Float64, ComplexF64}}
    if svdinit
        U, L, V = svd(X)
        Ubar = U[:, 1:K]
        Vbar = V[:, 1:K] * diagm(L[1:K])
    else
        if F <: Real
            Ubar = rand(Normal(), (D, K))
            Vbar = rand(Normal(), (T, K))
        else
            Ubar = [rand(ComplexNormal()) for i in 1:D, j in 1:K]
            Vbar = [rand(ComplexNormal()) for i in 1:T, j in 1:K]
        end
    end
    Σbar_U = diagm(zeros(F, K))
    Σbar_V = diagm(zeros(F, K))
    C_V = diagm(ones(F, K))
    s² = 1.0
    return SVDParams(Ubar, Vbar, Σbar_U, Σbar_V, C_V, s²)
end

function loglik(X :: Matrix{ComplexF64}, sp :: SVDParams, hp :: SVDHyperParams)
    M = sp.Ubar * sp.Vbar'
    return real(-(hp.D * hp.T) * log(π * sp.s²) - tr((X - M)' * (X - M)) / sp.s²)
end

function loglik(X :: Matrix{Float64}, sp :: SVDParams, hp :: SVDHyperParams)
    M = sp.Ubar * sp.Vbar'
    return -(hp.D * hp.T) * log(2 * π * sp.s²) / 2 - tr((X - M)' * (X - M)) / (2 * sp.s²)
end

function freeenergy(X :: Matrix{ComplexF64}, sp :: SVDParams, hp :: SVDHyperParams)
    # compute variational free energy
    D, T, K = hp.D, hp.T, hp.K

    fenergy = - K * real(logdet(sp.Σbar_U)) + K * D * log(hp.σ²_U) -
              K * real(logdet(sp.Σbar_V)) + T * real(logdet(sp.C_V)) -
              K * (D + T) + hp.σ²_U ^ (-1) * tr(hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) +
              tr(sp.C_V ^ (-1) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar)) + D * T * log(π * sp.s²) +
              (tr(X' * X) - 2 * tr(real(X' * sp.Ubar * sp.Vbar')) +
               tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))) / sp.s²
    return real(fenergy)
end

function freeenergy(X :: Matrix{Float64}, sp :: SVDParams, hp :: SVDHyperParams)
    # compute variational free energy
    D, T, K = hp.D, hp.T, hp.K

    fenergy = - K / 2 * logdet(sp.Σbar_U) + K * D / 2 * log(hp.σ²_U) -
              K / 2 * logdet(sp.Σbar_V) + T / 2 * logdet(sp.C_V) -
              K * (D + T) / 2 + (2 * hp.σ²_U) ^ (-1) * tr(hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) +
              1 / 2 * tr(sp.C_V ^ (-1) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar)) + D * T / 2 * log(2 * π * sp.s²) +
              (tr(X' * X) - 2 * tr(real(X' * sp.Ubar * sp.Vbar')) +
               tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))) / (2 * sp.s²)
    return fenergy
end


function update_Ubar!(X :: Matrix{F}, sp :: SVDParams) where {F <: Union{Float64, ComplexF64}}
    sp.Ubar = X * sp.Vbar * sp.Σbar_U / sp.s²
end

function update_Vbar!(X :: Matrix{F}, sp :: SVDParams) where {F <: Union{Float64, ComplexF64}}
    sp.Vbar = X' * sp.Ubar * sp.Σbar_V / sp.s²
end

function update_Σbar_U!(sp :: SVDParams, hp :: SVDHyperParams; hasmissing = false)
    sp.Σbar_U = inv(hp.σ²_U ^ (-1) * I + (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) / sp.s²)
end

function update_Σbar_V!(sp :: SVDParams, hp :: SVDHyperParams; hasmissing = false)
    sp.Σbar_V = inv(sp.C_V ^ (-1) + (hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) / sp.s²)
end

function update_C_V!(sp :: SVDParams, hp :: SVDHyperParams)
    for k in 1:hp.K
        @views sp.C_V[k, k] = norm(sp.Vbar[:, k]) ^ 2 / hp.T + sp.Σbar_V[k, k]
    end
end

function update_s²!(X :: Matrix{F}, sp :: SVDParams, hp :: SVDHyperParams) where {F <: Union{Float64, ComplexF64}}
    numer = norm(X) ^ 2 - 2 * tr(real.(X' * sp.Ubar * sp.Vbar')) +
            tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))
    sp.s² = real(numer / (hp.D * hp.T))
end

function bayesiansvd(X :: Matrix{F}, K :: Int64, n_iter :: Int64;
                     σ²_U :: Float64 = 1e5, σ²_V :: Float64 = 1e5,
                     svdinit :: Bool = false, learn_C_V :: Bool = true) where {F <: Union{Float64, ComplexF64}}

    # Bayesian SVD without missing values
    # --- arguments ---
    # X: data matrix (D×T Complex Matrix)
    # K: truncation rank (integer)
    # n_iter: the number of iterations of variational inference (integer)

    D, T = size(X)
    sp = init_svdparams(X, D, T, K, svdinit = svdinit)
    hp = SVDHyperParams(σ²_U, D, T, K)

    logliks = Vector{Float64}(undef, n_iter + 1)
    logliks[1] = loglik(X, sp, hp)
    freeenergies = Vector{Float64}(undef, n_iter + 1)
    freeenergies[1] = freeenergy(X, sp, hp)

    if !learn_C_V
        sp.C_V = diagm(repeat([σ²_V], K))
    end

    progress = Progress(n_iter)
    for i in 1:n_iter
        update_Σbar_U!(sp, hp)
        update_Σbar_V!(sp, hp)
        if i % 2 == 0
            update_Ubar!(X, sp)
        else
            update_Vbar!(X, sp)
        end
        if learn_C_V
            update_C_V!(sp, hp)
        end
        update_s²!(X, sp, hp)

        freeenergies[i + 1] = freeenergy(X, sp, hp)
        logliks[i + 1] = loglik(X, sp, hp)
        next!(progress)
    end
    return sp, hp, freeenergies, logliks
end
