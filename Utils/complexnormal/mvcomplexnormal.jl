using Distributions
using LinearAlgebra
using PDMats
using SparseArrays
using Base: rand
include("./AbstractComplexDist.jl")

struct MvComplexNormal{T <: AbstractVector{<: Union{Real, Complex}},
                       S <: AbstractMatrix} <: AbstractComplexDist
    μ :: T
    Γ :: S
    function MvComplexNormal(μ :: T,
                             Γ :: S;
                             check_posdef :: Bool = true,
                             check_hermitian :: Bool = true) where {T, S}
        if length(μ) == 1
            if !(length(μ) == size(Γ)[1])
                error("ERROR: dimensional mismatch")
            end
            if check_posdef && !(isreal(Γ[1]) && real(Γ[1]) > 0)
                error("ERROR: Γ must be positive definite.")
            end
        else
            if !(length(μ) == size(Γ)[1] == size(Γ)[2])
                error("ERROR: dimensional mismatch")
            end
            if check_hermitian && !ishermitian(Γ)
                error("ERROR: Γ must be hermitian.")
            end
            if check_posdef && !isposdef(Γ)
                error("ERROR: Γ must be positive definite.")
            end
        end
        return new{T, S}(μ, Γ)
    end
end

# outer constructors
function MvComplexNormal(μ :: AbstractVector{<: Union{Real, Complex}})
    d = length(μ)
    Γ = Matrix{Float64}(I, d, d)
    return MvComplexNormal(μ, Γ)
end

function MvComplexNormal(μ :: AbstractVector{<: Union{Real, Complex}}, σ :: Float64)
    d = length(μ)
    Γ = spdiagm(0 => fill(σ^2, d))
    return MvComplexNormal(μ, Γ)
end

function Base.rand(mvcn :: MvComplexNormal{T, S},
                   n :: Int64) where {T <: AbstractVector{<: Union{Real, Complex}},
                                      S <: AbstractMatrix{<: Union{Real, Complex}}}
    μ = mvcn.μ
    Γ = mvcn.Γ
    d = length(μ)

    μ_multivariate = vcat(real.(μ), imag.(μ))
    Γ_multivariate = 0.5 .* [real.(Γ) -imag.(Γ); imag.(Γ) real.(Γ)]
    Γ_multivariate = 0.5 .* (Γ_multivariate .+ Γ_multivariate')

    if !isposdef(Γ_multivariate)
        eig_Γ = eigen(Γ_multivariate)
        replaced_eigvals = deepcopy(eig_Γ.values)
        replaced_eigvals[replaced_eigvals .< 1e-10] .= 1e-10
        Γ_multivariate = eig_Γ.vectors * diagm(0 => replaced_eigvals) * eig_Γ.vectors'
        Γ_multivariate = Symmetric(Γ_multivariate)
    end

    if isa(Γ_multivariate, SparseMatrixCSC)
        x_multivariate = rand(MvNormal(μ_multivariate, PDSparseMat(Γ_multivariate)), n)
    else
        x_multivariate = rand(MvNormal(μ_multivariate, Γ_multivariate), n)
    end
    x_reshaped = reshape(x_multivariate[1:d, :] .+ im * x_multivariate[d+1:end, :], (d, n))
    return x_reshaped
end

function Base.rand(mvcn :: MvComplexNormal)
    return Base.rand(mvcn, 1)[:, 1]
end

function Distributions.loglikelihood(mvcn :: MvComplexNormal,
                                     z :: Vector)
    n = length(z)
    log_const = -real(log(π) + logdet(mvcn.Γ))
    log_exp = -real((z - mvcn.μ)' * mvcn.Γ ^ (-1) * (z - mvcn.μ))
    return log_const + log_exp
end
