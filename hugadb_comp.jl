### handle HuGaDB Dataset
### see details in: https://github.com/romanchereshnev/HuGaDB
### Chereshnev R., Kertész-Farkas A. (2018) HuGaDB: Human Gait Database for Activity Recognition
### from Wearable Inertial Sensor Networks. In: van der Aalst W. et al. (eds)
### Analysis of Images, Social Networks and Texts. AIST 2017.
### Lecture Notes in Computer Science, vol 10716. Springer, Cham
### https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12


using Plots
using Plots.PlotMeasures
using CSV
using DataFrames
using JLD2
using Random
using MKLSparse
using LaTeXStrings
using Colors

include("variationalsvd.jl")
include("bayesiandmd.jl")

outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end

fname = "data/Data/HuGaDB_v1_bicycling_01_01.txt"
datarange = collect(901:1050)
#fname = "data/Data/HuGaDB_v1_walking_08_00.txt"
#datarange = collect(101:250)
df = CSV.read(fname, delim = '\t', header = 4)

X = Matrix{Union{ComplexF64}}(df[datarange, r"a"])'
X = X[1:(end - 1), :]
heatmap(real.(X), dpi = 300)

include("DMD.jl")

#K = 2 # for walking
K = 3 # for cycling

sp, vhp, freeenergies, logliks_svd = bayesiansvd(X, K, 50,
                                                 σ²_U = 1e10, svdinit = true,
                                                 learn_C_V = false)

p1 = plot(logliks_svd, lw = 2, title = "log likelihood", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy", legend = :none)
p = plot(p1, p2)

U, L, V = svd(X)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

D, T = size(X)

X1 = real.(X)
X2 = real.(UK * LK * VK')
X3 = real.(sp.Ubar * sp.Vbar')
cmin, cmax = findmin(hcat(X1, X3))[1], findmax(hcat(X1, X3))[1]
p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "t", ylabel = "x")
p2 = heatmap(1:T, 1:D, X2, clims = (cmin, cmax),
             title = "SVD",
             xlabel = "t", ylabel = "x")
p3 = heatmap(1:T, 1:D, X3, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "t", ylabel = "x")
p = plot(p1, p2, p3)


n_iter = 5000
hp = DMDHyperParams(sp, vhp)
dp_ary, logliks = run_sampling(X, hp, n_iter)
@save "$outdir/mcmc_cycling_missing_K3.jld2" X X_missing dp_ary hp
plot(logliks)


λs = Array{ComplexF64, 2}(undef, hp.K, n_iter)
Ws = Array{ComplexF64, 3}(undef, hp.K, hp.K, n_iter)
for k in 1:hp.K
    map(i -> λs[k, i] = dp_ary[i].λ[k], 1:n_iter)
    for l in 1:hp.K
        map(i -> Ws[k, l, i] = dp_ary[i].W[k, l], 1:n_iter)
    end
end
p1 = plot(real.(transpose(λs)), title = "traceplot of eigvals (real)")
for k in 1:hp.K
    hline!(real.([naive_dp.λ[k]]), lw = 2, label = "l$k")
end
p2 = plot(imag.(transpose(λs)), title = "traceplot of eigvals (imag)")
for k in 1:hp.K
    hline!(imag.([naive_dp.λ[k]]), lw = 2, label = "l$k")
end
plot(p1, p2)
