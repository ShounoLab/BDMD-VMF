# Variational Inference for Bayesian SVD model 
using LinearAlgebra
using Distributions
using ProgressMeter

include("ComplexNormal.jl")

mutable struct SVDParams
    Ubar :: Matrix{Complex{Float64}}
    Vbar :: Matrix{Complex{Float64}}
    Σbar_U :: Array{Complex{Float64}, 3}
    Σbar_V :: Array{Complex{Float64}, 3}
    Σbar_U_inv :: Array{Complex{Float64}, 3}
    Σbar_V_inv :: Array{Complex{Float64}, 3}
    C_V :: Matrix{Float64}
    s² :: Float64
end

struct SVDHyperParams
    C_U :: Matrix{Float64}
    D :: Int64
    T :: Int64
    K :: Int64
end

function SVDParams(Ubar :: Matrix{Complex{Float64}}, Vbar :: Matrix{Complex{Float64}},
                   Σbar_U :: Array{Complex{Float64}, 3}, Σbar_V :: Array{Complex{Float64}, 3},
                   C_V :: Matrix{Complex{Float64}}, s² :: Float64)
    # outer constructor

    D = size(Ubar)[1]
    T = size(Vbar)[1]
    Σbar_U_inv = Array{Complex{Float64}, 3}(undef, (D, K, K))
    Σbar_V_inv = Array{Complex{Float64}, 3}(undef, (T, K, K))

    println(K)

    map(d -> Σbar_U_inv[d, :, :] .= inv(Σbar_U[d, :, :]), 1:D)
    map(t -> Σbar_V_inv[t, :, :] .= inv(Σbar_V[t, :, :]), 1:T)

    return SVDParams(Ubar, Vbar, Σbar_U, Σbar_V, Σbar_U_inv, Σbar_V_inv, C_V, s²)
end

function init_svdparams(X :: Matrix{Union{Missing, Complex{Float64}}}, D :: Int64, T :: Int64, K :: Int64)
    Ubar = [rand(ComplexNormal()) for i in 1:D, j in 1:K]
    Vbar = [rand(ComplexNormal()) for i in 1:T, j in 1:K]

    Σbar_U = Array{Complex{Float64}, 3}(undef, (D, K, K))
    Σbar_V = Array{Complex{Float64}, 3}(undef, (T, K, K))

    [@views Σbar_U[d, :, :] = diagm(ones(Complex{Float64}, K)) for d in 1:D]
    [@views Σbar_V[t, :, :] = diagm(ones(Complex{Float64}, K)) for t in 1:T]

    C_V = diagm(ones(Complex{Float64}, K))
    s² = 1.0
    return SVDParams(Ubar, Vbar, Σbar_U, Σbar_V, C_V, s²)
end

function loglik(X :: Matrix{Union{Missing, Complex{Float64}}},
                sp :: SVDParams, hp :: SVDHyperParams)
    indices = findall(.!ismissing.(X))
    N = length(indices)

    sum_in_l = 0.0
    for inds in indices
        d, t = inds[1], inds[2]
        sum_in_l += abs(X[inds] - transpose(sp.Ubar[d, :]) * conj(sp.Vbar[t, :])) ^ 2
    end
    return -real(N * log(π * sp.s²) + sum_in_l / sp.s²)
end

function freeenergy(X :: Matrix{Union{Missing, Complex{Float64}}},
                    sp :: SVDParams, hp :: SVDHyperParams)
    # compute variational free energy
    D, T, K = hp.D, hp.T, hp.K
    indices = findall(.!ismissing.(X))
    N = length(indices)

    sum_in_f = 0.0
    for inds in indices
        d, t = inds[1], inds[2]


        sum_in_f += X[inds] - 2 * real(X[inds] * transpose(sp.Vbar[t, :]) * conj(sp.Ubar[d, :])) +
                    tr((sp.Σbar_U[d, :, :] + conj(sp.Ubar[d, :]) * transpose(sp.Ubar[d, :])) *
                       (sp.Σbar_V[t, :, :] + conj(sp.Vbar[t, :]) * transpose(sp.Vbar[t, :])))
    end

    logdet_Σbar_U = sum([log(real(det(sp.Σbar_U[d, :, :]))) for d in 1:D])
    logdet_Σbar_V = sum([log(real(det(sp.Σbar_V[t, :, :]))) for t in 1:T])

    sum_Σbar_U = reshape(sum(sp.Σbar_U, dims = 1), (K, K))
    sum_Σbar_V = reshape(sum(sp.Σbar_V, dims = 1), (K, K))

    fenergy = N * log(π * sp.s²) + D * log(det(hp.C_U)) + T * log(det(sp.C_V)) -
              logdet_Σbar_U - logdet_Σbar_V - (D + T) * K +
              tr(hp.C_U ^ (-1) * (sum_Σbar_U + sp.Ubar' * sp.Ubar)) +
              tr(sp.C_V ^ (-1) * (sum_Σbar_V + sp.Vbar' * sp.Vbar)) +
              sum_in_f / sp.s²

    return real(fenergy)
end

function update_Ubar!(X :: Matrix{Union{Missing, Complex{Float64}}},
                      sp :: SVDParams, hp ::SVDHyperParams)
    for d in 1:hp.D
        sum_vx = zeros(Complex{Float64}, (1, hp.K))
        for t in 1:hp.T
            if !ismissing(X[d, t])
                sum_vx += transpose(sp.Vbar[t, :]) * X[d, t]
            end
        end
        sp.Ubar[d, :] = sum_vx * sp.Σbar_U[d, :, :] / sp.s²
    end
end

function update_Vbar!(X :: Matrix{Union{Missing, Complex{Float64}}},
                      sp :: SVDParams, hp ::SVDHyperParams)
    for t in 1:hp.T
        sum_ux = zeros(Complex{Float64}, (1, hp.K))
        for d in 1:hp.D
            if !ismissing(X[d, t])
                @views sum_ux += transpose(sp.Ubar[d, :]) * X[d, t]'
            end
        end
        @views sp.Vbar[t, :] = sum_ux * sp.Σbar_V[t, :, :] / sp.s²
    end
end

function update_Σbar_U!(X :: Matrix{Union{Missing, Complex{Float64}}},
                        sp :: SVDParams, hp :: SVDHyperParams)
    for d in 1:hp.D
        sum_vv = zeros(Complex{Float64}, (hp.K, hp.K))
        for t in 1:hp.T
            if !ismissing(X[d, t])
                @views sum_vv += conj(sp.Vbar[t, :]) * transpose(sp.Vbar[t, :]) + sp.Σbar_V[t, :, :]
            end
        end
        sp.Σbar_U_inv[d, :, :] = inv(hp.C_U) + sum_vv ./ sp.s²
        sp.Σbar_U[d, :, :] = inv(sp.Σbar_U_inv[d, :, :])
    end
end

function update_Σbar_V!(X :: Matrix{Union{Missing, Complex{Float64}}},
                        sp :: SVDParams, hp :: SVDHyperParams)
    for t in 1:hp.T
        sum_uu = zeros(Complex{Float64}, (hp.K, hp.K))
        for d in 1:hp.D
            if !ismissing(X[d, t])
                @views sum_uu += conj(sp.Ubar[d, :]) * transpose(sp.Ubar[d, :]) + sp.Σbar_U[d, :, :]
            end
        end
        sp.Σbar_V_inv[t, :, :] = inv(sp.C_V) + sum_uu ./ sp.s²
        sp.Σbar_V[t, :, :] = inv(sp.Σbar_V_inv[t, :, :])
    end
end

function update_C_V!(sp :: SVDParams, hp :: SVDHyperParams)
    for k in 1:hp.K
        @views sp.C_V[k, k] = real(norm(sp.Vbar[:, k]) ^ 2 +
                                   sum(sp.Σbar_V[:, k, k])) / hp.T
    end
end

function update_s²!(X :: Matrix{Union{Missing, Complex{Float64}}},
                    sp :: SVDParams, hp :: SVDHyperParams)
    indices = findall(.!ismissing.(X))
    numer = 0.0
    for inds in indices
        d, t = inds[1], inds[2]
        numer += abs(X[inds]) ^ 2 - 2 * real(X[inds] * transpose(sp.Ubar[d, :]) * conj(sp.Vbar[t, :])) +
                 tr((sp.Σbar_U[d, :, :] + conj(sp.Ubar[d, :]) * transpose(sp.Ubar[d, :])) *
                    (sp.Σbar_V[t, :, :] + conj(sp.Vbar[t, :]) * transpose(sp.Vbar[t, :])))
    end
    sp.s² = real(numer / length(indices))
end

function bayesiansvd(X :: Matrix{Union{Complex{Float64}, Missing}}, K :: Int64, n_iter :: Int64;
                     σ²_U :: Float64 = 1e5, σ²_V :: Float64 = 1e5,
                     learn_C_V :: Bool = true)
    # Bayesian SVD with missing values
    # --- arguments ---
    # X: data matrix (D×T Complex Matrix)
    # K: truncation rank (integer)
    # n_iter: the number of iterations of variational inference (integer)

    D, T = size(X)
    sp = init_svdparams(X, D, T, K)
    hp = SVDHyperParams(diagm(fill(σ²_U, K)), D, T, K)

    logliks = Vector{Float64}(undef, n_iter + 1)
    logliks[1] = loglik(X, sp, hp)
    freeenergies = Vector{Float64}(undef, n_iter + 1)
    freeenergies[1] = freeenergy(X, sp, hp)

    if !learn_C_V
        sp.C_V = diagm(fill(σ²_V, K))
    end

    progress = Progress(n_iter)
    for i in 1:n_iter
        update_Σbar_U!(X, sp, hp)
        update_Σbar_V!(X, sp, hp)
        if i % 2 == 0
            update_Ubar!(X, sp, hp)
        else
            update_Vbar!(X, sp, hp)
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

