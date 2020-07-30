using Plots
using Plots.PlotMeasures
using JLD2
using Random
using MKLSparse
using LaTeXStrings
using Colors

include("variationalsvd_missingvals.jl")
include("bayesiandmd_missingvals.jl")

Random.seed!(1234)

D = 32
T = 128
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
X_missing = make_missing(X, prob = 0.9)

t_ary = collect(range(0, 4 * pi, length = T))
#tmag_ary = collect(range(0, 4 * pi, length = Tmag))
d_ary = collect(range(-5, 5, length = D))
p1 = heatmap(t_ary, d_ary, real.(X))
#p2 = heatmap(tmag_ary, d_ary, real.(X_missing))
p2 = heatmap(t_ary, d_ary, real.(X_missing), dpi = 200, colorbar = :none,
             ticks = :none, axis = false)
plot(p1, p2)
plot(p2, xlabel = "", axis = :none)
savefig(p2, "oscillator_90.png")


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
@load "$outdir/mcmc_oscillator_missing_0.8.jld2" X X_missing dp_ary hp sp vhp


dp_map = map_bdmd(dp_ary, hp, 3000)
X_res = reconstruct_map(dp_map, hp)

X1 = real.(X)
X2 = real.(X_missing)
X3 = real.(X_res)
cmin, cmax = findmin(hcat(X1, X3))[1], findmax(hcat(X1, X3))[1]
cmin, cmax = findmin(hcat(X1))[1], findmax(hcat(X1))[1]
p1 = heatmap(1:hp.T, 1:hp.D, X1, clims = (cmin, cmax),
             title = "original", xlabel = "t", ylabel = "x")
p2 = heatmap(1:hp.T, 1:hp.D, X2, clims = (cmin, cmax),
             title = "missing", xlabel = "t", ylabel = "x")
p3 = heatmap(1:hp.T, 1:hp.D, X3, clims = (cmin, cmax),
             title = "reconst (MAP)", xlabel = "t", ylabel = "x")
p = plot(p1, p2, p3)
p = plot(p1, p2)

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
p = plot(p1, p2, dpi = 150)
savefig(p, "$outdir/oscillator_eigvals.png")

W_naive = transpose(transpose(naive_dp.Φ) .* naive_dp.b)
Ws = reshape(Ws, (2 * hp.K, n_iter))

p1 = plot(real.(transpose(Ws)), title = "traceplot of modes (real)")
for k in 1:hp.K
    for l in 1:hp.K
        hline!(real.([W_naive[k, l]]), lw = 2, label = "W$(k * (k - 1) + l)")
    end
end
p2 = plot(imag.(transpose(Ws)), title = "traceplot of modes (imag)")
for k in 1:hp.K
    for l in 1:hp.K
        hline!(imag.([W_naive[k, l]]), lw = 2, label = "W$(k * (k - 1) + l)")
    end
end
p = plot(p1, p2, dpi = 150)
savefig(p, "$outdir/oscillator_Ws.png")

X_preds = reconstruct_bdmd_missingvals(dp_ary, hp, sp, 5000, 2000)

X1 = real.(X)
X2 = imag.(X)
cmin, cmax = minimum(hcat(X1, X2)), maximum(hcat(X1, X2))
t_ary = collect(range(0, 4 * pi, length = hp.T))
d_ary = collect(range(-5, 5, length = hp.D))
for prob in 0.0:0.1:0.9
    @load "$outdir/mcmc_oscillator_missing_$prob.jld2" X X_missing dp_ary hp
    @load "$outdir/mcmc_oscillator_reconst_$prob.jld2" X_preds X_map
    X_mean = reconstruct_map(mean_bdmd(dp_ary, hp, 3000), hp)
    pr = Int64(prob * 100)
    p = heatmap(t_ary, d_ary, real.(X_mean), clims = (cmin, cmax),
                dpi = 300, colorbar = :none, xaxis = false, yaxis = false)
    savefig(p, "$outdir/oscillator_meanreconst_real_$pr.pdf")
    p = heatmap(t_ary, d_ary, imag.(X_mean), clims = (cmin, cmax),
                dpi = 300, colorbar = :none, xaxis = false, yaxis = false)
    savefig(p, "$outdir/oscillator_meanreconst_imag_$pr.pdf")
end

d, t = 2, 5
p1 = histogram(real.(X_preds[d, t, :]), normalize = true)
vline!([real(X[d, t])], lw = 5)
p2 = histogram(imag.(X_preds[d, t, :]), normalize = true)
vline!([imag(X[d, t])], lw = 5)
plot(p1, p2, dpi = 150)

X_quantiles_real, X_quantiles_imag = get_quantiles(X_preds, interval = 0.8)

d = 10
p = plot(real.(X[d, :]), dpi = 300, ribbon = (real.(X[d, :]) .- X_quantiles_real[d, :, 1],
                                          X_quantiles_real[d, :, 2] .- real.(X[d, :])),
         line = (:dot, 2), label = "original", legend = :none,
         linecolor = :deepskyblue4, fillcolor = :slategray)
plot!(real.(X_missing[d, :]), line = (:solid, 3), label = "observed",
      markercolor = :royalblue3, markeralpha = 0.5, seriestype = :scatter)
plot(p)


X1 = real.(X)
X2 = imag.(X)
cmin, cmax = minimum(hcat(X1, X2)), maximum(hcat(X1, X2))
t_ary = collect(range(0, 4 * pi, length = hp.T))
d_ary = collect(range(-5, 5, length = hp.D))
p = heatmap(t_ary, d_ary, real.(X), clims = (cmin, cmax),
            xlabel = L"\tau_t", ylabel = L"x_d", dpi = 300,
            xtickfontsize = 10,
            ytickfontsize = 10,
            xguidefontsize = 18,
            yguidefontsize = 18,
            legendfontsize = 18, margin = 0px)
savefig(p, "$outdir/oscillator_real.pdf")
p = heatmap(t_ary, d_ary, imag.(X), clims = (cmin, cmax),
            xlabel = L"\tau_t", ylabel = L"x_d", dpi = 300,
            xtickfontsize = 10,
            ytickfontsize = 10,
            xguidefontsize = 12,
            yguidefontsize = 12, margin = 0px)
savefig(p, "$outdir/oscillator_imag.pdf")
