using Plots
using StatsBase
using JLD
include("variationalsvd_missingvals.jl")
include("bayesiandmd_missingvals.jl")

D = 16
T = 64
K = 2


### load data ###
include("Utils/toydata.jl")
gen_oscillator("toydata_oscillator.csv", D, T, 5e-2, seed = 123)
X = CSV.read("data/toydata_oscillator.csv")
X = Matrix(transpose(Matrix{Union{Missing, Complex{Float64}}}(parse.(Complex{Float64}, X))))

### drop missing data ###
X_missing = deepcopy(X)
missing_prob = 0.3
missing_inds = rand(Bernoulli(1 - missing_prob), size(X))
X_missing[findall(iszero.(missing_inds))] .= missing

t_ary = collect(range(0, 4 * pi, length = T))
d_ary = collect(range(-5, 5, length = D))
p1 = heatmap(t_ary, d_ary, real.(X))
p2 = heatmap(t_ary, d_ary, real.(X_missing))
plot(p1, p2)

### variational SVD ###
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X_missing, K, 100,
                                                 σ²_U = 1 / D, learn_C_V = true)

p1 = plot(logliks_svd, lw = 2, title = "log likelihood", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy", legend = :none)
p = plot(p1, p2)

X1 = real.(X)
X2 = real.(X_missing)
X3 = real.(sp.Ubar * sp.Vbar')
cmin, cmax = findmin(hcat(X1, X3))[1], findmax(hcat(X1, X3))[1]
p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "t", ylabel = "x")
p2 = heatmap(1:T, 1:D, X2, clims = (cmin, cmax),
             title = "missing", xlabel = "t", ylabel = "x")
p3 = heatmap(1:T, 1:D, X3, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "t", ylabel = "x")
p4 = heatmap(1:T, 1:D, abs.(X1 .- X3), clims = (cmin, cmax),
             title = "abs(diff)",
             xlabel = "t", ylabel = "x")
p = plot(p1, p2, p3, p4)


### Bayesian DMD ###
n_iter = 10000
hp = BDMDHyperParams(sp, vhp)
dp_ary, logliks = run_sampling(X, hp, n_iter)

λ1 = [dp_ary[i].λ[1] for i in 1:n_iter]
λ2 = [dp_ary[i].λ[2] for i in 1:n_iter]

save("./mcmc_oscillator_missing.jld", "dp_ary", dp_ary, "hp", hp)
