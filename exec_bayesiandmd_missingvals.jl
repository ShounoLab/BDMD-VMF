using Plots
using StatsBase
using JLD2
using Random
using MKLSparse

include("variationalsvd_missingvals.jl")
include("bayesiandmd_missingvals.jl")

Random.seed!(1234)

D = 24
T = 64
K = 2

### load data ###
include("Utils/toydata.jl")
gen_oscillator("toydata_oscillator.csv", D, T, 5e-2, seed = 123)
X = CSV.read("data/toydata_oscillator.csv")
X = Matrix(transpose(Matrix{Union{Missing, Complex{Float64}}}(parse.(Complex{Float64}, X))))

### drop missing data ###
#sr_mag = 2
#Tmag = sr_mag * T
#X_missing = make_missing(X, sr_mag = sr_mag)
X_missing = make_missing(X, prob = 0.5)

t_ary = collect(range(0, 4 * pi, length = T))
#tmag_ary = collect(range(0, 4 * pi, length = Tmag))
d_ary = collect(range(-5, 5, length = D))
p1 = heatmap(t_ary, d_ary, real.(X))
#p2 = heatmap(tmag_ary, d_ary, real.(X_missing))
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
p = plot(p1, p2, p3)


### Bayesian DMD ###
n_iter = 10000
hp = BDMDHyperParams(sp, vhp)
dp_ary, logliks = run_sampling(X, hp, n_iter)
@save "$outdir/mcmc_oscillator_missing.jld2" X X_missing dp_ary hp

dp_map = map_bdmd(dp_ary, hp, 5000)
X_res = reconstruct_map(dp_map, hp)

# naive DMD
include("DMD.jl")
naive_dp = solve_dmd(X, K)


outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end

λs = Array{ComplexF64, 2}(undef, K, n_iter)
Ws = Array{ComplexF64, 3}(undef, K, K, n_iter)
for k in 1:K
    map(i -> λs[k, i] = dp_ary[i].λ[k], 1:n_iter)
    for l in 1:K
        map(i -> Ws[k, l, i] = dp_ary[i].W[k, l], 1:n_iter)
    end
end
p1 = plot(real.(transpose(λs)), title = "traceplot of eigvals (real)", label = ["lambda1, lambda2"])
#hline!(real.([naive_dp.λ[1], naive_dp.λ[2]]), lw = 2)
hline!(real.([naive_dp.λ[1]]), lw = 2)
hline!(real.([naive_dp.λ[2]]), lw = 2)
p2 = plot(imag.(transpose(λs)), title = "traceplot of eigvals (imag)")
#hline!(imag.([naive_dp.λ[1], naive_dp.λ[2]]), lw = 2)
hline!(imag.([naive_dp.λ[1]]), lw = 2)
hline!(imag.([naive_dp.λ[2]]), lw = 2)
p = plot(p1, p2, dpi = 300)
savefig(p, "$outdir/oscillator_eigvals.png")


p1 = heatmap(t_ary, d_ary, real.(X), title = "original (real)")
p2 = heatmap(t_ary, d_ary, imag.(X), title = "original (imag)")
#p2 = heatmap(tmag_ary, d_ary, real.(X_missing))
p3 = heatmap(t_ary, d_ary, real.(X_missing), title = "corrupted (real)")
p4 = heatmap(t_ary, d_ary, imag.(X_missing), title = "corrupted (imag)")
p = plot(p1, p2, p3, p4, dpi = 300)
savefig(p, "$outdir/oscillator_data.png")
