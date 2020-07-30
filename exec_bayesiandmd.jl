using Plots
using StatsBase
using JLD2
include("variationalsvd.jl")
include("bayesiandmd.jl")
include("DMD.jl")
include("Utils/NLSESimulations/pseudospectral.jl")


outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end



### Nonlinear Schrodinger Equation
Ngrids = 256 # Number of Fourier modes
L = 30.0 # Space period
Δt = 2π / 21
t_end = 2π
Nsteps = round(Int, t_end / Δt)

config = NLSESettings(Nsteps, Δt, t_end, Ngrids, L)

# initial state of the wave function
ψ₀ = 2.0 * sech.(config.gridpoints)

#result = SSFM(ψ₀, config)
result = PseudoSpectral(ψ₀, config)
X = result.ψ

D, T = size(result.ψ)
t_ary = collect(0:config.Δt:config.t_end)[1:end - 1]

p = heatmap(t_ary, config.gridpoints, abs.(X),
            title = "original", xlabel = "t", ylabel = "x")
savefig(p, "$outdir/NLSE_pseudospectral.pdf")

#K = 8
K = 4
naive_dp = solve_dmd(X, K)

X_reconst_dmd = reconstruct(t_ary, t_ary, naive_dp)

p1, p2, p3 = heatmap(abs.(X)), heatmap(abs.(X_reconst_dmd)), heatmap(abs.(X .- X_reconst_dmd))
plot(p1, p2, p3)


### Bayesian DMD ###
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X, K, 200, σ²_U = 1 / D, svdinit = true,
                                                 learn_C_V = true)

p1 = plot(logliks_svd, lw = 2, title = "log likelihood", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy", legend = :none)
p = plot(p1, p2)

U, L, V = svd(X)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

X1 = abs.(X)
X2 = abs.(UK * LK * VK')
X3 = abs.(sp.Ubar * sp.Vbar')
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

plot([real(dp_ary[i].λ[k]) for i in 1:length(dp_ary), k in 1:K])
plot(logliks)

Ws = Array{ComplexF64, 3}(undef, hp.K, hp.K, mc.n_iter)
for k in 1:hp.K
    for l in 1:hp.K
        map(i -> Ws[k, l, i] = dp_ary[i].W[k, l], 1:mc.n_iter)
    end
end

plot([real(Ws[4, k, i]) for i in 1:length(dp_ary), k in 1:K])

plot([dp_ary[i].σ² for i in 1:length(dp_ary)])


dp_mean = mean_bdmd(dp_ary, hp, mc)
X_meanreconst_bdmd = reconstruct_pointest(dp_mean, hp)


X1, X2 = abs.(X_reconst_dmd), abs.(X_meanreconst_bdmd)
X3, X4 = abs.(X - X_reconst_dmd), abs.(X - X_meanreconst_bdmd)
cmin, cmax = minimum(hcat(X1, X2)), maximum(hcat(X1, X2))
p1 = heatmap(X1, title = "DMD", clims = (cmin, cmax))
p2 = heatmap(X2, title = "BDMD-VMF (EAP)", clims = (cmin, cmax))
cmin, cmax = minimum(hcat(X3, X4)), maximum(hcat(X3, X4))
p3 = heatmap(X3, title = "DMD (abs. error)", clims = (cmin, cmax))
p4 = heatmap(X4, title = "BDMD-VMF (abs. error)", clims = (cmin, cmax))
p = plot(p1, p2, p3, p4, dpi = 200)
savefig(p, "$outdir/BDMD_reconst.png")


p1 = visualize_eigvals(naive_dp.λ, title = "eigvals by DMD")
p2 = visualize_eigvals(dp_mean.λ, title = "eigvals by BDMD-VMF")
p = plot(p1, p2, dpi = 200)
savefig(p, "$outdir/DMD_BDMD_eigvals.png")


X_preds = reconstruct_bdmd(dp_ary, hp, sp, mc)

@save "$outdir/mcmc_nlse.jld2" X X_preds dp_ary logliks hp mc
#@load "$outdir/mcmc_nlse.jld2" X X_preds dp_ary logliks hp mc
