### handle HuGaDB Dataset
### see details in: https://github.com/romanchereshnev/HuGaDB
### Chereshnev R., Kertész-Farkas A. (2018) HuGaDB: Human Gait Database for Activity Recognition
### from Wearable Inertial Sensor Networks. In: van der Aalst W. et al. (eds)
### Analysis of Images, Social Networks and Texts. AIST 2017.
### Lecture Notes in Computer Science, vol 10716. Springer, Cham
### https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12


using Plots
using Plots.PlotMeasures
using DelimitedFiles
using CSV
using DataFrames
using JLD2
using Random
using Colors
include("variationalsvd_missingvals.jl")
include("bayesiandmd_missingvals.jl")

outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end

fname = "data/Data/HuGaDB_v1_bicycling_01_01.txt"
datarange = collect(901:1050)
#fname = "data/Data/HuGaDB_v1_walking_08_00.txt"
#datarange = collect(101:250)
df = CSV.read(fname, delim = '\t', header = 4)

X = Matrix(Matrix{Union{Missing, Complex{Float64}}}(df[datarange, r"g"])')
# linear transformation into [-1, 1]
minx, maxx = minimum(real.(X)), maximum(real.(X))
X = real.(2 / (maxx - minx) .* (X .- minx) .- 1)
X = Matrix{Union{Missing, ComplexF64}}(X)


K = 2

include("DMD.jl")
naive_dp = solve_dmd(X, K)

# make masked data
X_missing = deepcopy(X)
X_missing[9:end, 76:end] .= missing
X_missing[1:12, 1:60] .= missing

writedlm("$outdir/hugadb_original.csv", real.(X), ",")
writedlm("$outdir/hugadb_missing.csv", real.(X_missing), ",")
#heatmap(real.(X_missing), dpi = 300)
#include("Utils/toydata.jl")
#X_missing = make_missing(X, prob = 0.5)
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X_missing, K, 100,
                                                 σ²_U = 1e10, σ²_V = 1.0,
                                                 learn_C_V = false,
                                                 showprogress = true)

p1 = plot(logliks_svd, lw = 2, title = "log likelihood", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy", legend = :none)
p = plot(p1, p2)

U, L, V = svd(X)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

D, T = size(X)

X1 = real.(X)
X2 = real.(X_missing)
X3 = real.(UK * LK * VK')
X4 = real.(sp.Ubar * sp.Vbar')
cmin, cmax = findmin(hcat(X1, X3))[1], findmax(hcat(X1, X3))[1]
p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "t", ylabel = "x")
p2 = heatmap(1:T, 1:D, X2, clims = (cmin, cmax),
             title = "missing", xlabel = "t", ylabel = "x")
p3 = heatmap(1:T, 1:D, X3, clims = (cmin, cmax),
             title = "SVD",
             xlabel = "t", ylabel = "x")
p4 = heatmap(1:T, 1:D, X4, clims = (cmin, cmax),
             title = "variational SVD",
             xlabel = "t", ylabel = "x")
p = plot(p1, p2, p3, p4)


n_iter = 5000
mc = MCMCConfig(n_iter, 3000, 1e-2, 1e-1, 1e-2)
hp = BDMDHyperParams(sp, vhp)
dp_ary, logliks = run_sampling(X_missing, hp, mc)
@save "$outdir/mcmc_cycling_maskmissing2_K2.jld2" X X_missing dp_ary hp sp vhp

X_mean = reconstruct_map(mean_bdmd(dp_ary, hp, mc.burnin), hp)
X_preds = reconstruct(dp_ary, hp, sp, 5000, mc.burnin)
@save "$outdir/mcmc_cycling_renconst_maskmissing2.jld2" X_preds X_mean



@load "$outdir/mcmc_cycling_maskmissing2_K2.jld2" X X_missing dp_ary hp sp vhp
@load "$outdir/mcmc_cycling_renconst_maskmissing2.jld2" X_preds X_mean
df_stan_0025 = CSV.read("df_0025.csv")[:, 2:end]
df_stan_mean = CSV.read("df_mean.csv")[:, 2:end]
df_stan_0975 = CSV.read("df_0975.csv")[:, 2:end]

X_quantiles_real, X_quantiles_imag = get_quantiles(X_preds, interval = 0.95)
X_quantiles_stan = Array{Float64}(undef, size(X_quantiles_real))
X_quantiles_stan[:, :, 1] .= Matrix(df_stan_0025)
X_quantiles_stan[:, :, 2] .= Matrix(df_stan_0975)

# d = 5, 10, 14, 18
d = 14
p = plot(real.(X[d, :]), dpi = 300, ribbon = (real.(X[d, :]) .- X_quantiles_stan[d, :, 1],
                                          X_quantiles_stan[d, :, 2] .- real.(X[d, :])),
         line = (:dot, 4), label = "original", legend = :none,
         linecolor = :deepskyblue4, fillcolor = :darkslategray, fillalpha = 0.3,
         xtickfontsize = 14,
         ytickfontsize = 14)
plot!(real.(X[d, :]), dpi = 300, ribbon = (real.(X[d, :]) .- X_quantiles_real[d, :, 1],
                                           X_quantiles_real[d, :, 2] .- real.(X[d, :])),
      line = (:dot, 4), label = "original", legend = :none,
      linecolor = :deepskyblue4, fillcolor = :brown, fillalpha = 0.3)
plot!(real.(X_missing[d, :]), line = (:solid, 3), label = "observed",
      markercolor = :royalblue3, markeralpha = 0.5,
      markersize = 6, markerstrokewidth = 2,
      seriestype = :scatter)
p = plot(p, dpi = 300)
savefig(p, "$outdir/cycling_$(string(names(df[:, r"g"])[d]))_95.pdf")
