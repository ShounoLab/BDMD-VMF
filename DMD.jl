using DataFrames
using LinearAlgebra

mutable struct DMDParams
    # λ <n_modes>: eigenvalues of Koopman Operator
    # W <n_data × n_modes>: Koopman modes
    # Φ <n_modes × n_modes>: eigenvectors of subspace dynamics
    # b <n_modes × 1>: amplitudes

    n_data :: Int64
    n_datadims :: Int64
    n_modes :: Int64
    λ :: Vector{Complex{Float64}}
    W :: Matrix{Complex{Float64}}
    Φ :: Matrix{Complex{Float64}}
    b :: Vector{Complex{Float64}}
end

function solve_dmd(X :: AbstractMatrix, n_modes :: Int64;
                   exact :: Bool = false)
    X₀ = X[:, 1:(end - 1)]
    X₁ = X[:, 2:end]

    n_data = size(X)[1] - 1
    n_datadims = size(X)[2]

    U, s, V = svd(X₀)
    Uₖ = U[:, 1:n_modes]
    Σₖ = diagm(s[1:n_modes])
    Vₖ = V[:, 1:n_modes]

    Atilde = Uₖ' * X₁ * Vₖ * Σₖ ^ (-1)
    λ, Φ = eigen(Atilde)

    if exact
        dmdmode = X₁ * Vₖ * Σₖ ^ (-1) * Φ
    else
        dmdmode = Uₖ * Φ
    end

    b_ary = dmdmode \ X₀[:, 1]
    return DMDParams(n_data, n_datadims, n_modes, λ, dmdmode, Φ, b_ary)
end

function reconstruct(original_time :: Vector{Float64},
                     t_ary :: Vector{Float64}, dp :: DMDParams)
    #Δt = t_ary[2] - t_ary[1]
    Δt = original_time[2] - original_time[1]
    Λc = diagm(log.(dp.λ)) / Δt

    reconstructed_mat = Matrix{Complex{Float64}}(undef, (length(t_ary), dp.n_datadims))
    for (i, t) in enumerate(t_ary)
        reconstructed_mat[i, :] = dp.w * exp(Λc * t) * dp.amplitudes
    end
    return reconstructed_mat
end
