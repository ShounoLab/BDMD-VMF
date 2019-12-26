using Distributions
using Base: rand
using LinearAlgebra
include("./AbstractComplexDist.jl")

struct ComplexNormal{T <: Complex, S <: Real} <: AbstractComplexDist
    μ :: T
    σ :: S
    function ComplexNormal(μ :: T, σ :: S) where {T, S}
        if (σ <= 0.0)
            error("ERROR: σ must be > 0.")
        end
        return new{T, S}(μ, σ)
    end
end

# outer constructors
ComplexNormal() = ComplexNormal(0.0 + 0im, 1.0)
ComplexNormal(μ :: Complex) = ComplexNormal(μ, 1.0)

function Base.rand(cn :: ComplexNormal{T, S}, n :: Int64) where {T <: Complex,
                                                                 S <: Real}
    μ = cn.μ
    σ² = cn.σ ^ 2

    μ_bivariate = [real(μ), imag(μ)]
    σ²_bivariate = 0.5 * [σ² 0; 0 σ²]

    if !isposdef(σ²_bivariate)
        eig_σ² = eigen(σ²_bivariate)
        replaced_eigvals = deepcopy(eig_σ².values)
        replaced_eigvals[replaced_eigvals .<= 0] .= 1e-10
        σ²_bivariate = eig_σ².vectors * diagm(0 => replaced_eigvals) * eig_σ².vectors'
        σ²_bivariate = Symmetric(σ²_bivariate)
    end

    x_bivariate = rand(MvNormal(μ_bivariate, σ²_bivariate), n)
    return x_bivariate[1, :] .+ x_bivariate[2, :] .* im
end

function Base.rand(cn :: ComplexNormal)
    return Base.rand(cn, 1)[1]
end

function Distributions.loglikelihood(cn :: ComplexNormal, z :: Union{Real, Complex})
    log_const = - (log(π) + 2 * log(cn.σ))
    log_exp = - real((z - cn.μ)' * (z - cn.μ)) / cn.σ ^ 2
    return log_const + log_exp
end
