# Variational Inference for Bayesian SVD model

mutable struct SVDParams
    Ubar :: Matrix{Complex{Float64}}
    Vbar :: Matrix{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    Σbar_V :: Matrix{Complex{Float64}}
end

struct SVDHyperParams
    σ²_V :: Float64
end
