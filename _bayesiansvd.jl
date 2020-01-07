# Variational Inference for Bayesian SVD model

using LinearAlgebra
using Distributions
using ProgressMeter
using Plots

mutable struct SVDParams
    Ubar :: Matrix{Float64}
    Vbar :: Matrix{Float64}
    Σbar_U :: Matrix{Float64}
    Σbar_V :: Matrix{Float64}
    s² :: Float64
    σ²_V :: Vector{Float64}
end

struct SVDHyperParams
    σ²_U :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function init_svdparams(D :: Int64, T :: Int64, K :: Int64)
    #Ubar = vcat(I, zeros(Complex{Float64}, D - K, K))
    #Vbar = vcat(I, zeros(Complex{Float64}, T - K, K))
    Ubar = [randn() for i in 1:D, j in 1:K]
    Vbar = [randn() for i in 1:T, j in 1:K]
    Σbar_U = diagm(ones(K))
    Σbar_V = diagm(ones(K))
    s² = 1.0 + 0.0im
    return SVDParams(Ubar, Vbar, Σbar_U, Σbar_V, s²)
    #return SVDParams(U_K[:, 1:K], (V_K * diagm(L_K))[:, 1:K], Σbar_U, Σbar_V, s²)
end

function loglik(X :: Matrix{Float64}, sp :: SVDParams, hp :: SVDHyperParams)
    M = sp.Ubar * sp.Vbar'
    return -(hp.D * hp.T) * log(π) - hp.T * log(sp.s²) - tr((X - M)' * (X - M)) / sp.s²
end

function freeenergy(X :: Matrix{Float64}, sp :: SVDParams, hp :: SVDHyperParams)
    # compute variationa free energy
    D, T, K = hp.D, hp.T, hp.K

    fenergy = - K * log(real(det(sp.Σbar_U))) + K * D * log(hp.σ²_U) -
              K * log(real(det(sp.Σbar_V))) + K * T * log(det(diagm(sp.σ²_V))) -
              K * (D + T) + hp.σ²_U ^ (-1) * tr(hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) +
              diagm(sp.σ²_V .^ (-1)) * tr(hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) + D * T * log(sp.s²) +
              (tr(X' * X) - 2 * tr(real(X' * sp.Ubar * sp.Vbar')) +
               tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))) / sp.s²
    return real(fenergy)
end

function update_Ubar!(X :: Matrix{Float64}, sp :: SVDParams)
    sp.Ubar = X * sp.Vbar * sp.Σbar_U / sp.s²
end

function update_Vbar!(X :: Matrix{Float64}, sp :: SVDParams)
    sp.Vbar = X' * sp.Ubar * sp.Σbar_V / sp.s²
end

function update_Σbar_U!(sp :: SVDParams, hp :: SVDHyperParams)
    sp.Σbar_U = inv(hp.σ²_U ^ (-1) * I + (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) / sp.s²)
end

function update_Σbar_V!(sp :: SVDParams, hp :: SVDHyperParams)
    sp.Σbar_V = inv(diagm(hp.σ²_V .^ (-1)) + (hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) / sp.s²)
end

function update_s²!(X :: Matrix{Float64}, sp :: SVDParams, hp :: SVDHyperParams)
    numer = tr(X' * X - X' * sp.Ubar * sp.Vbar' - sp.Vbar * sp.Ubar' * X) +
            tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))
    sp.s² = real(numer / (hp.D * hp.T))
end

function update_σ

function bayesiansvd(X :: Matrix{Float64}, K :: Int64, n_iter :: Int64;
                     σ²_U :: Float64 = 1e5, σ²_V :: Float64 = 1e5)
    # X: data matrix (D×T Complex Matrix)
    # K: truncation rank (integer)
    # n_iter: the number of iterations of variational inference (integer)

    D, T = size(X)
    sp = init_svdparams(D, T, K)
    hp = SVDHyperParams(σ²_U, σ²_V, D, T, K)

    logliks = Vector{Float64}(undef, n_iter + 1)
    logliks[1] = loglik(X, sp, hp)
    freeenergies = Vector{Float64}(undef, n_iter + 1)
    freeenergies[1] = freeenergy(X, sp, hp)

    sp_ary = Vector{SVDParams}(undef, n_iter + 1)
    sp_ary[1] = deepcopy(sp)

    progress = Progress(n_iter)
    for i in 1:n_iter
        update_Ubar!(X, sp)
        update_Σbar_U!(sp, hp)
        update_Vbar!(X, sp)
        update_Σbar_V!(sp, hp)
        update_s²!(X, sp, hp)

        freeenergies[i + 1] = freeenergy(X, sp, hp)
        logliks[i + 1] = loglik(X, sp, hp)
        sp_ary[i + 1] = deepcopy(sp)
        next!(progress)
    end
    return sp_ary, hp, freeenergies, logliks
end

D = 3
T = 5
K = 2
#X = [rand(ComplexNormal(0im, 1)) for i in 1:D, j in 1:T]
X = reshape(collect(1:D*T) .* 1.0, (D, T))
U_K, L_K, V_K = svd(X)
U_K, L_K, V_K = U_K[:, 1:K], diagm(L_K[1:K]), V_K[:, 1:K]
U_K * L_K * V_K'

sp_ary, hp, freeenergies, logliks = bayesiansvd(X, K, 100, σ²_U = 1/D, σ²_V = 1e10)
plot(logliks)
plot(freeenergies)

p1 = plot(real.(U_K[:, 1:K]), lw = 2)
p2 = plot(real.(sp.Ubar), lw = 2)
plot(p1, p2)
heatmap(sp.Ubar * sp.Vbar')
