using Random
using Distributions
using DataFrames
using CSV

include("../ComplexNormal.jl")

function gen_data(fname :: String, T :: Int64, λ_true :: Vector, σ :: Float64;
                  seed :: Int64 = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    data = zeros(T, length(λ_true))
    for t in 1:T
        data[t, :] .= λ_true .^ (t-1)
    end
    data = data .+ rand(Normal(0, σ), size(data)...)

    data = DataFrame(data)

    dname = "./data"
    if !isdir(dname)
        mkdir(dname)
    end

    CSV.write(dname * "/" * fname, data)

    return nothing
end

function gen_oscillator(fname :: String, D :: Int64, T :: Int64, σ :: Float64;
                        seed :: Int64 = nothing)
    function f1(t :: Real, x :: Real)
        return 1.0 / cosh(x + 3) * exp(2.3im * t)
    end

    function f2(t :: Real, x :: Real)
        return 2.0 / cosh(x) * tanh(x) * exp(2.8im * t)
    end

    if !isnothing(seed)
        Random.seed!(seed)
    end

    xs = collect(range(-5, 5, length = D))
    ts = collect(range(0, 4 * pi, length = T))

    X1 = Matrix{Complex}(undef, (T, D))
    X2 = Matrix{Complex}(undef, (T, D))
    for (t, tt) in enumerate(ts)
        for (d, xd) in enumerate(xs)
            X1[t, d] = f1(tt, xd)
            X2[t, d] = f2(tt, xd)
        end
    end
    X = X1 .+ X2

    data = X .+ [rand(ComplexNormal(0.0 + 0.0im, σ)) for i in 1:size(X)[1], j in 1:size(X)[2]]
    data = DataFrame(data)

    dname = "./data"
    if !isdir(dname)
        mkdir(dname)
    end

    CSV.write(dname * "/" * fname, data)

    return nothing
end

function make_missing(X :: Matrix{Union{Missing, Complex{Float64}}};
                      prob :: Union{Nothing, Float64} = nothing,
                      sr_mag :: Union{Nothing, Int64} = nothing)
    if isnothing(sr_mag)
        X_missing = deepcopy(X)
        missing_inds = rand(Bernoulli(1 - prob), size(X))
        X_missing[findall(iszero.(missing_inds))] .= missing
        return X_missing
    else
        X_missing = Matrix{Union{Missing, Complex{Float64}}}(missing, size(X)[1], sr_mag * size(X)[2])
        for (i, t) in enumerate(1:sr_mag:(sr_mag * size(X)[2]))
            X_missing[:, t] .= X[:, i]
        end
        return X_missing
    end
end


