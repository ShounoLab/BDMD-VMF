using Plots
using RDatasets
using StatsBase
using Distributions
using Random

Random.seed!(123)

include("variationalsvd_missingvals.jl")
iris = dataset("datasets", "iris")
iris = Matrix(transpose(Matrix{Union{Missing, Complex{Float64}}}(iris[:, 1:4])))
iris_missing = deepcopy(iris)
K = 2
D, T = size(iris)

missing_prob = 0.3
missing_inds = rand(Bernoulli(1 - missing_prob), size(iris))
iris_missing[findall(iszero.(missing_inds))] .= missing

U, L, V = svd(iris)
Ubar = deepcopy(U[:, 1:K])
Vbar = deepcopy(V[:, 1:K] * diagm(L[1:K])')

sp, hp, freeenergies, logliks = bayesiansvd(iris_missing, K, 200, σ²_U = 1 / D,
                                            learn_C_V = true)

p1 = plot(logliks, lw = 2, title = "log likelihood (iris)", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy (iris)", legend = :none)
p = plot(p1, p2)
savefig(p, "iris_convergence.pdf")


X1 = real.(iris)
X2 = real.(iris_missing)
X3 = real.(sp.Ubar * sp.Vbar')
cmin, cmax = findmin(hcat(X1, X3))[1], findmax(hcat(X1, X3))[1]
p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "sample", ylabel = "feature")
p2 = heatmap(1:T, 1:D, X2, clims = (cmin, cmax),
             title = "missing", xlabel = "sample", ylabel = "feature")
p3 = heatmap(1:T, 1:D, X3, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "sample", ylabel = "feature")
p4 = heatmap(1:T, 1:D, X1 .- X3, clims = (cmin, cmax),
             title = "diff",
             xlabel = "sample", ylabel = "feature")
p = plot(p1, p2, p3, p4)
savefig(p, "iris_reconst.pdf")

p1 = scatter(real.(sp.Vbar[:, 1]), real.(sp.Vbar[:, 2]),
             title = "variational SVD", label = "latent data")
p2 = scatter(real.(V[:, 1]), real.(V[:, 2]),
             title = "complete data SVD", label = "projected data")
p = plot(p1, p2)
