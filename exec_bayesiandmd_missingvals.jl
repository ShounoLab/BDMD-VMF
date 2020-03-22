using Plots
using StatsBase
using JLD2
using Random
using MKLSparse

include("variationalsvd_missingvals.jl")
include("bayesiandmd_missingvals.jl")

Random.seed!(1234)

D = 16
T = 32
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
sp, vhp, freeenergies, logliks_svd = bayesiansvd(X_missing, K, 200,
                                                 σ²_U = 1e10, σ²_V = 1.0,
                                                 learn_C_V = false,
                                                 showprogress = true)

p1 = plot(logliks_svd, lw = 2, title = "log likelihood", legend = :none)
p2 = plot(freeenergies, lw = 2, title = "free energy", legend = :none)
p = plot(p1, p2)

U, L, V = svd(X)
UK, LK, VK = U[:, 1:K], diagm(L[1:K]), V[:, 1:K]

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


outdir = "output"

if !isdir(outdir)
    mkdir(outdir)
end

# naive DMD
include("DMD.jl")
naive_dp = solve_dmd(X, K)

### Bayesian DMD ###
include("bayesiandmd_missingvals.jl")
n_iter = 5000
hp = BDMDHyperParams(sp, vhp)
dp_ary, logliks = run_sampling(X_missing, hp, n_iter)
@save "$outdir/mcmc_oscillator_missing.jld2" X X_missing dp_ary hp

dp_map = map_bdmd(dp_ary, hp, 3000)
X_res = reconstruct_map(dp_map, hp)

p1 = heatmap(1:T, 1:D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "t", ylabel = "x")
p2 = heatmap(1:T, 1:D, real.(X_missing), clims = (cmin, cmax),
             title = "missing", xlabel = "t", ylabel = "x")
p3 = heatmap(1:T, 1:D, real.(X_res), clims = (cmin, cmax),
             title = "reconst (MAP)", xlabel = "t", ylabel = "x")
p = plot(p1, p2, p3)

λs = Array{ComplexF64, 2}(undef, K, n_iter)
Ws = Array{ComplexF64, 3}(undef, K, K, n_iter)
for k in 1:K
    map(i -> λs[k, i] = dp_ary[i].λ[k], 1:n_iter)
    for l in 1:K
        map(i -> Ws[k, l, i] = dp_ary[i].W[k, l], 1:n_iter)
    end
end
p1 = plot(real.(transpose(λs)), title = "traceplot of eigvals (real)")
for k in 1:K
    hline!(real.([naive_dp.λ[k]]), lw = 2, label = "l$k")
end
p2 = plot(imag.(transpose(λs)), title = "traceplot of eigvals (imag)")
for k in 1:K
    hline!(imag.([naive_dp.λ[k]]), lw = 2, label = "l$k")
end
p = plot(p1, p2, dpi = 150)
savefig(p, "$outdir/oscillator_eigvals.png")

W_naive = transpose(transpose(naive_dp.Φ) .* naive_dp.b)
Ws = reshape(Ws, (2 * K, n_iter))

p1 = plot(real.(transpose(Ws)), title = "traceplot of modes (real)")
for k in 1:K
    for l in 1:K
        hline!(real.([W_naive[k, l]]), lw = 2, label = "W$(k * (k - 1) + l)")
    end
end
p2 = plot(imag.(transpose(Ws)), title = "traceplot of modes (imag)")
for k in 1:K
    for l in 1:K
        hline!(imag.([W_naive[k, l]]), lw = 2, label = "W$(k * (k - 1) + l)")
    end
end
p = plot(p1, p2, dpi = 150)
savefig(p, "$outdir/oscillator_Ws.png")

X_preds = reconstruct(dp_ary, hp, sp, 5000, 2000)

d, t = 2, 5
p1 = histogram(real.(X_preds[d, t, :]), normalize = true)
vline!([real(X[d, t])], lw = 5)
p2 = histogram(imag.(X_preds[d, t, :]), normalize = true)
vline!([imag(X[d, t])], lw = 5)
plot(p1, p2, dpi = 150)

X_quantiles_real, X_quantiles_imag = get_quantiles(X_preds, interval = 0.50)

d = 2
p1 = plot(real.(X_missing[d, :]), dpi = 150, ribbon = (real.(X[d, :]) .- X_quantiles_real[d, :, 1],
                                          X_quantiles_real[d, :, 2] .- real.(X[d, :])))
p2 = plot(real.(X[d, :]), dpi = 150, ribbon = (real.(X[d, :]) .- X_quantiles_real[d, :, 1],
                                          X_quantiles_real[d, :, 2] .- real.(X[d, :])))
plot(p1, p2)
