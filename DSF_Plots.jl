import Plots as Plt
import BSplineKit as BSK
import CurveFit as CF
using Roots, DelimitedFiles, Printf, LaTeXStrings

# Reading data

data, head = readdlm("f_Bose1D.txt"; header = true)
z = data[:,1]
f = data[:,2]

# f(z) algebraic fit

Wz = length(z)÷10
p = CF.CurveFitProblem(log.(z[end-Wz:end]), log.(f[end-Wz:end]))
sol = CF.solve(p, CF.LinearCurveFitAlgorithm())
a = sol.u[1]
b = sol.u[2]
println("exposant f(z): $(-a)")

# f(z) lin-lin plot

Plt.plot(z, f, linestyle=:solid, linecolor=:red, linewidth=3)
Plt.plot!(xlabel=L"z = (|\omega|-c|k|)/\gamma_k", ylabel=L"f(z)", guidefontsize=16, tickfontsize=14, framestyle=:box, legend=false)
Plt.xlims!(0, 6)
Plt.ylims!(0, 1.02)

# f(z) log-log plot

iz0 = length(z)÷2+1
zfit = z[iz0+320:end]

Plt.plot!(z[iz0+1:end], f[iz0+1:end], subplot=2, linestyle=:solid, linecolor=:red, linewidth=3, inset=Plt.bbox(0.02, 0.03, 0.3, 0.3, :top, :right))
Plt.plot!(zfit, 0.8 .* zfit.^(-1/3), subplot=2, linestyle=:dot, linecolor=:black, linewidth=3)
Plt.plot!(subplot=2, xscale=:log, yscale=:log, xlabel=L"z", guidefontsize=16, tickfontsize=10, framestyle=:box, legend = false)
Plt.xticks!(([1e0, 1e1, 1e2, 1e3, 1e4], ["   10^{0}", "", "10^{2}", "", "10^{4}      "]), subplot=2)
Plt.yticks!(([0.1], ["10^{-1}"]), subplot=2)
Plt.annotate!((-0.15, 0.8), Plt.text(L"f(z)", 16, :center, rotation = 90), subplot=2)
Plt.annotate!((0.75, 0.65), Plt.text(L"\propto z^{-1/3}", 16, :center), subplot=2)
Plt.xlims!(1e0, 1e4, subplot=2)
Plt.ylims!(2e-2, 1.1e0, subplot=2)
Plt.savefig("fz.pdf")

# Dynamic structure factor
S = @. f/(z^2+f^2)

# Reading data

data, head = readdlm("S_KPZ.txt"; header = true)
y = data[:,1]
S_KPZ = data[:,2]

# Scaling parameter (0.5 = half-height, 1e-3 = tail)
t = 0.5

# Computing KPZ scaling factor

Sz = BSK.interpolate(z[iz0:end], S[iz0:end] .- t, BSK.BSplineOrder(4))
z0 = find_zero(Sz, 1-t)

iy0 = length(y)÷2+1
Sy = BSK.interpolate(y[iy0:end], S_KPZ[iy0:end] .- t, BSK.BSplineOrder(4))
y0 = find_zero(Sy, 1-t)

sf = z0/y0
println("KPZ scaling factor: $(sf)")

# S(z) algebraic fit

Wy = length(y)÷10

p = CF.CurveFitProblem(log.(z[end-Wz:end]), log.(S[end-Wz:end]))
sol = CF.solve(p, CF.LinearCurveFitAlgorithm())
a = sol.u[1]
b = sol.u[2]

p = CF.CurveFitProblem(log.(sf.*y[end-Wy:end]), log.(S_KPZ[end-Wy:end]))
sol = CF.solve(p, CF.LinearCurveFitAlgorithm())
a_KPZ = sol.u[1]
b_KPZ = sol.u[2]

println("exposant S(z): $(-a)")
println("exposant KPZ: $(-a_KPZ)")
println("prefacteur S(z): $(exp(b))")
println("prefacteur KPZ: $(exp(b_KPZ))")
println("ratio: $(abs(exp(b)-exp(b_KPZ))/exp(b))")

# S(z) lin-lin plot

Plt.plot(z, S, linestyle=:solid, linecolor=:red, linewidth=3, label="1D Bose gas")
Plt.plot!(sf.*y, S_KPZ, linestyle=:dash, linecolor=:darkcyan, linewidth=3, label="KPZ")
Plt.plot!(xlabel=L"z = (|\omega|-c|k|)/\gamma_k", ylabel=L"S(z)", guidefontsize=16, tickfontsize=14, framestyle=:box, legend_position=:topleft, legend_font_pointsize=12)
Plt.xlims!(-6, 6)
Plt.ylims!(-0.02, 1.02)

# S(z) log-log plot

zfit = z[iz0+250:end]

Plt.plot!(z, S, subplot=2, linestyle=:solid, linecolor=:red, linewidth=3, label="1D Bose gas", inset = Plt.bbox(0.02, 0.03, 0.29, 0.3, :top, :right))
Plt.plot!(sf.*y, S_KPZ, subplot=2, linestyle=:dash, linecolor=:darkcyan, linewidth=3, label="KPZ")
Plt.plot!(zfit, 1.4 .* zfit.^(-7/3), subplot=2, linestyle=:dot, linecolor=:black, linewidth=3)
Plt.plot!(subplot=2, xscale=:log, yscale=:log, xlabel=L"z", guidefontsize=16, tickfontsize=10, framestyle=:box, guidefontvalign=:top, legend=false)
Plt.xticks!(([1.0, 10.0, 100.0], ["     10^{0}", "10^{1}", "10^{2}      "]), subplot=2)
Plt.yticks!(([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], ["10^{-5}", "", "10^{-3}", "", "", ""]), subplot=2)
Plt.annotate!((-0.15, 0.8), Plt.text(L"S(z)", 16, :center, rotation=90), subplot=2)
Plt.annotate!((0.75, 0.65), Plt.text(L"\propto z^{-7/3}", 16, :center), subplot=2)
Plt.xlims!(1e0, 1e2, subplot=2)
Plt.ylims!(1e-5, 1.01e0, subplot=2)
Plt.savefig("Sz.pdf")
