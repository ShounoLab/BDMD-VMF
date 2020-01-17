using Plots
using RDatasets
using StatsBase

include("variationalsvd.jl")
iris = dataset("datasets", "iris")
iris = Matrix(transpose(Matrix{Complex{Float64}}(iris[:, 1:4])))
K = 2
D, T = size(iris)

sp, hp, freeenergies, logliks = bayesiansvd(iris, K, 100, σ²_U = 1 / D, svdinit = true,
                                            learn_C_V = true)

p1 = plot(logliks, lw = 2, title = "log likelihood (iris)", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy (iris)", legend = :none)
p = plot(p1, p2)
savefig(p, "iris_convergence.pdf")

U, L, V = svd(iris)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

X1 = real.(iris)
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
savefig(p, "iris_reconst.pdf")

X1 = real.(VK * LK)
X2 = real.(sp.Vbar)
p1 = scatter(X1[:, 1], X1[:, 2], title = "naive SVD")
p2 = scatter(X2[:, 1], X2[:, 2], title = "variational SVD")
p = plot(p1, p2)
savefig(p, "iris_projected.pdf")


mtcars = dataset("datasets", "mtcars")
delete!(mtcars, [:Model, :VS, :AM])
mtcars = Matrix(transpose(Matrix{Complex{Float64}}(mtcars)))
K = 2
D, T = size(mtcars)

sp, hp, freeenergies, logliks = bayesiansvd(mtcars, K, 100, σ²_U = 1 / D, svdinit = true,
                                            learn_C_V = true)

p1 = plot(logliks[2:end], lw = 2, title = "log likelihood (mtcars)", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy (mtcars)", legend = :none)
p = plot(p1, p2)
savefig(p, "mtcars_convergence.pdf")

U, L, V = svd(mtcars)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

function unit_normalize(X :: Matrix{Float64})
    return StatsBase.transform(fit(UnitRangeTransform, X), X)
end

X1 = unit_normalize(real.(mtcars))
X2 = unit_normalize(real.(UK * LK * VK'))
X3 = unit_normalize(real.(sp.Ubar * sp.Vbar'))
cmin, cmax = findmin(hcat(X1, X2, X3))[1], findmax(hcat(X1, X2, X3))[1]
p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "sample", ylabel = "feature")
p2 = heatmap(1:T, 1:D, X2, clims = (cmin, cmax),
             title = "naive SVD", xlabel = "sample", ylabel = "feature")
p3 = heatmap(1:T, 1:D, X3, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "sample", ylabel = "feature")
p = plot(p1, p2, p3)
savefig(p, "mtcars_reconst.pdf")

X1 = real.(VK * LK)
X2 = real.(sp.Vbar)
p1 = scatter(X1[:, 1], X1[:, 2], title = "naive SVD")
p2 = scatter(X2[:, 1], X2[:, 2], title = "variational SVD")
p = plot(p1, p2)
savefig(p, "mtcars_projected.pdf")
