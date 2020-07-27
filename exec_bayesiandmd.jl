using Plots
using StatsBase
using MAT
using JLD2
include("variationalsvd.jl")
include("bayesiandmd.jl")
include("DMD.jl")


outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end



### Nonlinear Schrodinger Equation
vars_nlse = matread("data/NLSE.mat")
X = vars_nlse["X"][193:320, :]
t_ary = reshape(vars_nlse["t"], :)
d_ary = reshape(vars_nlse["xi"], :)[193:320]
heatmap(t_ary, d_ary, abs.(X))
D, T = Int(vars_nlse["n"]), Int(vars_nlse["slices"])

#K = 8
K = 4
naive_dp = solve_dmd(X, K)
plot(real.(naive_dp.λ), imag(naive_dp.λ), seriestype = :scatter)

X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)

p1, p2, p3 = heatmap(real.(X)), heatmap(real.(X_reconst_dmd)), heatmap(abs.(real.(X) .- real.(X_reconst_dmd)))
plot(p1, p2, p3)


### Bayesian DMD ###
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X, K, 100, σ²_U = 1 / D, svdinit = true,
                                                 learn_C_V = true)

p1 = plot(logliks_svd, lw = 2, title = "log likelihood", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy", legend = :none)
p = plot(p1, p2)

U, L, V = svd(X)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

X1 = real.(X)
X2 = real.(UK * LK * VK')
X3 = real.(sp.Ubar * sp.Vbar')
cmin, cmax = findmin(hcat(X1, X2, X3))[1], findmax(hcat(X1, X2, X3))[1]
p1 = heatmap(X1, clims = (cmin, cmax),
             title = "original", xlabel = "sample", ylabel = "feature")
p2 = heatmap(X2, clims = (cmin, cmax),
             title = "naive SVD", xlabel = "sample", ylabel = "feature")
p3 = heatmap(X3, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "sample", ylabel = "feature")
p = plot(p1, p2, p3)


### Bayesian DMD ###
mc = MCMCConfig(7500, 3000, 1e-2, 1e-1, 1e-2)
hp = BDMDHyperParams(sp.Ubar, sp.Σbar_U, 1e5, 1e5, 0.01, 0.01, D, T, K)
dp_ary, logliks = run_sampling(X, hp, mc)

plot([imag(dp_ary[i].λ[k]) for i in 1:length(dp_ary), k in 1:K])
plot(logliks)
λs_bdmd = [dp_ary[end].λ[k] for k in 1:K]

Ws = Array{ComplexF64, 3}(undef, hp.K, hp.K, mc.n_iter)
for k in 1:hp.K
    for l in 1:hp.K
        map(i -> Ws[k, l, i] = dp_ary[i].W[k, l], 1:mc.n_iter)
    end
end

plot([real(Ws[4, k, i]) for i in 1:length(dp_ary), k in 1:K])

plot([dp_ary[i].σ² for i in 1:length(dp_ary)])


# reconstruction using the final state of MCMC
X_reconst_bdmd = Matrix{ComplexF64}(undef, size(X))
λ, W, σ² = dp_ary[end].λ, dp_ary[end].W, dp_ary[end].σ²
for t in 1:hp.T
    gₜ = W * (λ .^ t)
    Gₜ = (gₜ * gₜ' / σ² + hp.Σbar_U ^ (-1)) ^ (-1)
    σₜ² = real(σ² * (1 - gₜ' * Gₜ * gₜ) ^ (-1))
    xₜ = σₜ² / σ² * hp.Ubar * hp.Σbar_U ^ (-1) * Gₜ * gₜ
    X_reconst_bdmd[:, t] .= xₜ
end

@save "$outdir/mcmc_nlse.jld2" X dp_ary logliks hp mc
