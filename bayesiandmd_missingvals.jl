using Distributions
using ProgressMeter
using SparseArrays

mutable struct BDMDParams
    λ :: Vector{Complex{Float64}}
    W :: Matrix{Complex{Float64}}
    σ² :: Float64
end

struct BDMDHyperParams
    vecUbar :: Vector{Complex{Float64}}
    Σbar_U :: Matrix{Complex{Float64}}
    Γbar_U :: Matrix{Complex{Float64}}
    Σbar_U_inv :: Matrix{Complex{Float64}}
    Γbar_U_inv :: Matrix{Complex{Float64}}
    σ²_λ :: Float64
    σ²_w :: Float64
    α :: Float64
    β :: Float64
    D :: Int64
    T :: Int64
    K :: Int64
end

function BDMDHyperParams(sp :: SVDParams, shp :: SVDHyperParams;
                         σ²_λ :: Float64 = 1e5, σ²_w :: Float64 = 1e5,
                         α :: Float64 = 0.01, β :: Float64 = 0.01)
    # outer constructor
    C = commutation_matrix(shp.D, shp.K)
    Σbar_U = zeros(Complex{Float64}, (shp.D * shp.K, shp.D * shp.K))
    Σbar_U_inv = zeros(Complex{Float64}, (shp.D * shp.K, shp.D * shp.K))

    map(d -> Σbar_U[((d - 1) * K + 1):(d * K), ((d - 1) * K + 1):(d * K)] =
             deepcopy(sp.Σbar_U[d, :, :]), 1:shp.D)
    map(d -> Σbar_U_inv[((d - 1) * K + 1):(d * K), ((d - 1) * K + 1):(d * K)] =
        deepcopy(sp.Σbar_U_inv[d, :, :]), 1:shp.D)

    Γᵤ⁻¹ = C' * transpose(Σbar_U_inv) * C
    Γᵤ = inv(Γᵤ⁻¹)
    return DMDHyperParams(vec(sp.Ubar), Σbar_U, Γᵤ, Σbar_U_inv, Γᵤ⁻¹,
                          σ²_λ, σ²_w, α, β,
                          shp.D, shp.T, shp.K)
end

function commutation_matrix(M :: Int64, N :: Int64)
    # get commutation matrix of vectrized matrix
    # see Def.2.1 in Magnus and Neudecker (1979) for details

    A = reshape(1:(M * N), (M, N))'
    A = vec(A)

    return diagm(ones(Int64, M * N))[A, :]
end

function sparsechol(A :: Matrix{Complex{Float64}})
    return cholesky(sparse(Hermitian(A)))
end

function logdetΣ(Σ :: Matrix{Complex{Float64}})
    return logdet(sparsechol(Σ))
end

function chol_inv(A :: Matrix{Complex{Float64}})
    L = Matrix(sparse(sparsechol(A).L))
    invL = inv(L)
    return invL' * invL
end

function loglik(X :: Matrix{Union{Missing, Complex{Float64}}},
                dp :: BDMDParams, hp :: BDMDHyperParams)
    I_D = diagm(ones(hp.D))
    G = zeros(Complex{Float64}, hp.K, hp.T)
    map(t -> G[:, t] = dp.W * (dp.λ .^ (t - 1)), 1:hp.T)
    Σᵤ⁻¹ = Hermitian(kron(transpose(G * G'), I_D) ./ dp.σ² + hp.Γbar_U_inv)
    Σᵤ = chol_inv(Σᵤ⁻¹)

    #Σₓ⁻¹ = I ./ dp.σ² - kron(transpose(G), I_D) * Σᵤ * kron(conj(G), I_D)
    Σₓ⁻¹ = - kron(transpose(G), I_D) * Σᵤ * kron(conj(G), I_D)
    map(i -> Σₓ⁻¹[i, i] += inv(dp.σ²), 1:(hp.D * hp.T))

    # inversion of Σₓ⁻¹ : Woodbury formula
    #Σₓ = I * dp.σ² + dp.σ² ^ 2 * kron(transpose(G), I_D) *
    #                 inv(Hermitian(Σᵤ⁻¹ - dp.σ² .* kron(transpose(G * G'), I_D))) *
    #                 kron(conj(G), I_D)
    Σₓ = dp.σ² ^ 2 * kron(transpose(G), I_D) *
                     chol_inv(Σᵤ⁻¹ - dp.σ² .* kron(transpose(G * G'), I_D)) *
                     kron(conj(G), I_D)
    map(i -> Σₓ[i, i] += dp.σ², 1:(hp.D * hp.T))

    inv(Σᵤ⁻¹ - dp.σ² .* kron(transpose(G * G'), I_D))

    vecXbar = Σₓ * kron(transpose(G), I_D) *
              Σᵤ * hp.Γbar_U_inv * hp.vecUbar

    # use non-missing entries to compute log likelihood
    vecX = vec(X)
    vecX = vec(X_missing)
    missvec = .!ismissing.(vecX)
    vecX_com = vecX[missvec]
    vecXbar_com = vecXbar[missvec]
    Σₓ_com = Σₓ[missvec, missvec]
    Σₓ⁻¹_com = Σₓ⁻¹[missvec, missvec]

    logL = -real(length(vecX_com) * log(π) + logdetΣ(Σₓ_com) +
                (vecX_com - vecXbar_com)' * Σₓ⁻¹_com * (vecX_com - vecXbar_com))
    return logL
end

function logprior(dp :: BDMDParams, hp :: BDMDHyperParams)
    logp = 0.0
    for k in 1:hp.K
        logp += loglikelihood(ComplexNormal(0.0im, √(hp.σ²_λ)), dp.λ[k])
        logp += loglikelihood(MvComplexNormal(zeros(Complex{Float64}, hp.K),
                                              √(hp.σ²_w)),
                              dp.W[:, k])
    end
    logp += logpdf(InverseGamma(hp.α, hp.β), dp.σ²)
    return logp
end

function metropolis!(X :: Matrix{Union{Missing, Complex{Float64}}},
                     dp :: BDMDParams, dp_cand :: BDMDParams, hp :: BDMDHyperParams)
    logp_orig = loglik(X, dp, hp) + logprior(dp, hp)
    logp_cand = loglik(X, dp_cand, hp) + logprior(dp_cand, hp)

    logr = logp_cand - logp_orig
    if logr > 0
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    elseif logr > log(rand())
        dp.λ, dp.W, dp.σ² = dp_cand.λ, dp_cand.W, dp_cand.σ²
    end
end

function metropolis_λ!(X :: Matrix{Union{Missing, Complex{Float64}}},
                       dp :: BDMDParams, hp :: BDMDHyperParams)
    σₚᵣₒₚ = 1e-2
    for k in 1:hp.K
        dp_cand = deepcopy(dp)
        dp_cand.λ[k] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
        metropolis!(X, dp, dp_cand, hp)
    end
end

function metropolis_W!(X :: Matrix{Union{Missing, Complex{Float64}}},
                       dp :: BDMDParams, hp :: BDMDHyperParams)
    σₚᵣₒₚ = 0.1
    @views for k in 1:hp.K
        for l in 1:hp.K
            dp_cand = deepcopy(dp)
            dp_cand.W[k, l] += rand(ComplexNormal(0.0im, σₚᵣₒₚ))
            metropolis!(X, dp, dp_cand, hp)
        end
    end
end

function metropolis_σ²!(X :: Matrix{Union{Missing, Complex{Float64}}},
                        dp :: BDMDParams, hp :: BDMDHyperParams)
    σₚᵣₒₚ = 1e-2
    dp_cand = deepcopy(dp)
    dp_cand.σ² += rand(Normal(0.0, σₚᵣₒₚ))
    if dp_cand.σ² > 0
        metropolis!(X, dp, dp_cand, hp)
    end
end

function init_dmdparams(hp :: BDMDHyperParams)
    λ = ones(Complex{Float64}, hp.K)
    W = zeros(Complex{Float64}, (hp.K, hp.K))
    σ² = 1.0
    return BDMDParams(λ, W, σ²)
end

function run_sampling(X :: Matrix{Union{Missing, Complex{Float64}}},
                      hp :: BDMDHyperParams, n_iter :: Int64)
    dp = init_dmdparams(hp)

    dp_ary = Vector{BDMDParams}(undef, n_iter)
    logliks = Vector{Float64}(undef, n_iter)

    progress = Progress(n_iter)
    for i in 1:n_iter
        metropolis_W!(X, dp, hp)
        metropolis_λ!(X, dp, hp)
        metropolis_σ²!(X, dp, hp)

        dp_ary[i] = deepcopy(dp)
        logliks[i] = loglik(X, dp, hp)
        next!(progress)
    end

    return dp_ary, logliks
end
