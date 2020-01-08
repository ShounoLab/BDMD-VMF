# Variational Inference for Bayesian SVD model

using LinearAlgebra
using Distributions
using ProgressMeter
using Plots
using RDatasets

mutable struct SVDParams
    Ubar :: Matrix{Complex{Float64}}
    Vbar :: Matrix{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    Σbar_V :: Matrix{Complex{Float64}}
    C_V :: Matrix{Complex{Float64}}
    s² :: Float64
end

struct SVDHyperParams
    σ²_U :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function init_svdparams(D :: Int64, T :: Int64, K :: Int64)
    U, L, V = svd(iris)
    Ubar = U[:, 1:K]
    Vbar = V[:, 1:K] * diagm(L[1:K])
    #Ubar = [rand(ComplexNormal()) for i in 1:D, j in 1:K]
    #Vbar = [rand(ComplexNormal()) for i in 1:T, j in 1:K]
    Σbar_U = diagm(zeros(Complex{Float64}, K))
    Σbar_V = diagm(zeros(Complex{Float64}, K))
    C_V = diagm(ones(Complex{Float64}, K))
    s² = 1.0
    return SVDParams(Ubar, Vbar, Σbar_U, Σbar_V, C_V, s²)
end

function loglik(X :: Matrix{Complex{Float64}}, sp :: SVDParams, hp :: SVDHyperParams)
    M = sp.Ubar * sp.Vbar'
    return -(hp.D * hp.T) * log(π) - hp.T * log(sp.s²) - tr((X - M)' * (X - M)) / sp.s²
end

function freeenergy(X :: Matrix{Complex{Float64}}, sp :: SVDParams, hp :: SVDHyperParams)
    # compute variationa free energy
    D, T, K = hp.D, hp.T, hp.K

    fenergy = - K * log(real(det(sp.Σbar_U))) + K * D * log(hp.σ²_U) -
              K * log(real(det(sp.Σbar_V))) + T * log(real(det(sp.C_V))) -
              K * (D + T) + hp.σ²_U ^ (-1) * tr(hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) +
              tr(sp.C_V ^ (-1) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar)) + D * T * log(sp.s²) +
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
    sp.Σbar_U = inv(hp.σ²_U ^ (-1) * I + (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar) / sp.s²)
end

function update_Σbar_V!(sp :: SVDParams, hp :: SVDHyperParams)
    sp.Σbar_V = inv(sp.C_V ^ (-1) + (hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) / sp.s²)
end

function update_C_V!(sp :: SVDParams, hp :: SVDHyperParams)
    for k in 1:hp.K
        @views sp.C_V[k, k] = norm(sp.Vbar[:, k]) ^ 2 / hp.T + sp.Σbar_V[k, k]
    end
end

function update_s²!(X :: Matrix{Complex{Float64}}, sp :: SVDParams, hp :: SVDHyperParams)
    numer = tr(X' * X - X' * sp.Ubar * sp.Vbar' - sp.Vbar * sp.Ubar' * X) +
            tr((hp.D * sp.Σbar_U + sp.Ubar' * sp.Ubar) * (hp.T * sp.Σbar_V + sp.Vbar' * sp.Vbar))
    sp.s² = real(numer / (hp.D * hp.T))
end

function bayesiansvd(X :: Matrix{Complex{Float64}}, K :: Int64, n_iter :: Int64;
                     σ²_U :: Float64 = 1e5)
    # X: data matrix (D×T Complex Matrix)
    # K: truncation rank (integer)
    # n_iter: the number of iterations of variational inference (integer)

    D, T = size(X)
    sp = init_svdparams(D, T, K)
    hp = SVDHyperParams(σ²_U, D, T, K)

    logliks = Vector{Float64}(undef, n_iter + 1)
    logliks[1] = loglik(X, sp, hp)
    freeenergies = Vector{Float64}(undef, n_iter + 1)
    freeenergies[1] = freeenergy(X, sp, hp)

    sp_ary = Vector{SVDParams}(undef, n_iter + 1)
    sp_ary[1] = deepcopy(sp)

    progress = Progress(n_iter)
    for i in 1:n_iter
        update_Σbar_U!(sp, hp)
        update_Σbar_V!(sp, hp)
        update_Ubar!(X, sp)
        update_Vbar!(X, sp)
        update_C_V!(sp, hp)
        update_s²!(X, sp, hp)

        freeenergies[i + 1] = freeenergy(X, sp, hp)
        logliks[i + 1] = loglik(X, sp, hp)
        sp_ary[i + 1] = deepcopy(sp)
        next!(progress)
    end
    return sp_ary, hp, freeenergies, logliks
end

include("ComplexNormal.jl")

iris = dataset("datasets", "iris")
iris = Matrix(transpose(Matrix{Complex{Float64}}(iris[:, 1:4])))
K = 2
D, T = size(iris)

sp_ary, hp, freeenergies, logliks = bayesiansvd(iris, K, 100, σ²_U = 1 / D)

plot(logliks)
plot(freeenergies)

U, L, V = svd(iris)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

p1 = heatmap(1:T, 1:D, real.(iris))
p2 = heatmap(1:T, 1:D, real.(UK * LK * VK'))
p3 = heatmap(1:T, 1:D, real.(sp_ary[end].Ubar * sp_ary[end].Vbar'))
plot(p1, p2, p3)

p1 = scatter(real.(V[:, 1]), real.(V[:, 2]))
p2 = scatter(real.(sp_ary[end].Vbar[:, 1]), real.(sp_ary[end].Vbar[:, 2]))

hoge = [sp_ary[i].C_V[1, 1] for i in 1:1001]
plot(real.(hoge))
