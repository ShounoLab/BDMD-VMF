### handle HuGaDB Dataset
### see details in: https://github.com/romanchereshnev/HuGaDB
### Chereshnev R., Kertész-Farkas A. (2018) HuGaDB: Human Gait Database for Activity Recognition
### from Wearable Inertial Sensor Networks. In: van der Aalst W. et al. (eds)
### Analysis of Images, Social Networks and Texts. AIST 2017.
### Lecture Notes in Computer Science, vol 10716. Springer, Cham
### https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12


using Plots
using CSV
using DataFrames

fname = "data/Data/HuGaDB_v1_walking_08_00.txt"
df = CSV.read(fname, delim = '\t', header = 4)
plot(Matrix(df[101:300, r"a"]), dpi = 300)

include("DMD.jl")

X = Matrix{ComplexF64}(df)'
dp = solve_dmd(X, 7)
p1 = scatter(real.(dp.λ), imag.(dp.λ), xlabel = "real eigvals", ylabel = "imag eigvals")
dp = solve_dmd(X, 16)
p2 = scatter(real.(dp.λ), imag.(dp.λ), xlabel = "real eigvals", ylabel = "imag eigvals")
dp = solve_dmd(X, 24)
p3 = scatter(real.(dp.λ), imag.(dp.λ), xlabel = "real eigvals", ylabel = "imag eigvals")
dp = solve_dmd(X, 32)
p4 = scatter(real.(dp.λ), imag.(dp.λ), xlabel = "real eigvals", ylabel = "imag eigvals")
plot(p1, p2, p3, p4, dpi = 300)

