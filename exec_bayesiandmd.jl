using Plots
using StatsBase
include("variationalsvd.jl")
include("bayesiandmd.jl")

D = 32
T = 64
K = 2


### load data ###
include("Utils/toydata.jl")
gen_oscillator("toydata_oscillator.csv", D, T, 5e-2, seed = 123)
X = CSV.read("data/toydata_oscillator.csv")
X = Matrix(transpose(Matrix(parse.(Complex{Float64}, X))))

t_ary = collect(range(0, 4 * pi, length = T))
d_ary = collect(range(-5, 5, length = D))
heatmap(t_ary, d_ary, real.(X))


### variational SVD ###
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
p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "sample", ylabel = "feature")
p2 = heatmap(1:T, 1:D, X2, clims = (cmin, cmax),
             title = "naive SVD", xlabel = "sample", ylabel = "feature")
p3 = heatmap(1:T, 1:D, X3, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "sample", ylabel = "feature")
p = plot(p1, p2, p3)

X1 = real.(VK * LK)
X2 = real.(sp_ary[end].Vbar)
p1 = scatter(X1[:, 1], X1[:, 2], title = "naive SVD")
p2 = scatter(X2[:, 1], X2[:, 2], title = "variational SVD")
p = plot(p1, p2)


### Bayesian DMD ###
hp = DMDHyperParams(sp.Ubar, sp.Σbar_U, 1e5, 1e5, 0.01, 0.01, D, T, K)
dp_ary, logliks = run_sampling(X, hp, 100)
