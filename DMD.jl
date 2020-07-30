using DataFrames
using LinearAlgebra
using Plots

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

    n_data = size(X)[2]
    n_datadims = size(X)[1]

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

    reconstructed_mat = Matrix{Complex{Float64}}(undef, (dp.n_datadims, length(t_ary)))
    for (i, t) in enumerate(t_ary)
        reconstructed_mat[:, i] = dp.W * exp(Λc * t) * dp.b
    end
    return reconstructed_mat
end

function visualize_eigvals(λ :: Vector{ComplexF64}; title = "eigvals")
    circleshape = (sin.(range(0, 2π, length = 500)),  cos.(range(0, 2π, length = 500)))
    p = plot(real.(λ), imag.(λ), title = title,
             seriestype = :scatter, legend = false,
             xlabel = "Re", ylabel = "Im",
             xlims = (-1, 1), ylims = (-1, 1),
             aspect_ratio = 1,
             framestyle = :origin)
    plot!(Shape(circleshape...), lw = 2, linestyle = :dash, linecolor = :green,
          fill = false, legend = false, fillalpha = 0)
    return p
end

