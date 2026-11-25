import Plots as Plt
import BSplineKit as BSK
import QuadGK as QGK
using DelimitedFiles, Printf, LaTeXStrings

# Reading data
data, head = readdlm("gy-deriv.txt"; header = true)
y0 = Float64.(data[:,1])
∂²g = data[:,4]
idx = findall(x -> isa(x, AbstractFloat), ∂²g)
y = [y0[i] for i in idx]
f = [∂²g[i] for i in idx]./4

# Data checks
fy = BSK.interpolate(y, f, BSK.BSplineOrder(4))
I0 = first(QGK.quadgk(x -> fy(x), minimum(y), maximum(y)))
println("∫f(x)dx = $(I0)") # = 1
I1 = first(QGK.quadgk(x -> fy(x)*x^2, minimum(y), maximum(y)))
println("∫f(x)x²dx = $(I1)") # = 0.510523

# Plot f(y)
Plt.plot(y,f)
Plt.xlims!(-2.4,2.4)
Plt.ylims!(-0.02, 0.6)
Plt.plot!(xlabel=L"y", ylabel=L"f(y)", guidefontsize=12, tickfontsize=12, framestyle=:box, legend=false)
Plt.savefig("f_KPZ(y).pdf")

# k array
kmin = 1e-2
kmax = 1e4
Nk = 1024+1
klog = exp10.(range(log10(kmin), log10(kmax), length=(Nk-1)÷2))
k = vcat(-reverse(klog, dims=1), zeros(1), klog)

# Fourier transform
f_hat(k) = first(QGK.quadgk(x -> 2*fy(x)*cos(k*x), 0, maximum(y)))
@time fk = f_hat.(k)

# Plot fk(k)
Plt.plot(k, fk)
Plt.xlims!(0,6)
Plt.ylims!(-0.05, 1.05)
Plt.plot!(xlabel=L"k", ylabel=L"\hat f(k)", guidefontsize=12, tickfontsize=12, framestyle=:box, legend=false)
Plt.plot!(k, fk, subplot=2, framestyle=:box, legend=false, inset = Plt.bbox(0.04, 0.05, 0.4, 0.4, :top, :right))
Plt.xlims!(4, 8, subplot=2)
Plt.ylims!(-0.0015, 0.004, subplot=2)
Plt.savefig("fk_KPZ(k).pdf")

# z array
zmin = 1e-2
zmax = 6e2
Nz = 1024+1
zlog = exp10.(range(log10(zmin), log10(zmax), length=(Nz-1)÷2))
z = vcat(-reverse(zlog, dims=1), zeros(1), zlog)
i0 = Nz÷2+1

# h(ω) function
fk_itp = BSK.interpolate(k, fk, BSK.BSplineOrder(4))
h(ω) = first(QGK.quadgk(t -> 2*fk_itp(abs(t)^(2/3))*cos(ω*t), 0, kmax))
@time S_hat = h.(z)

open("S_KPZ.txt", "w") do io
    println(io, "z\tS(z)")
    for i in eachindex(z)
        @printf(io, "%.18e %.18e\n", z[i], S_hat[i]/S_hat[i0])
    end
end
