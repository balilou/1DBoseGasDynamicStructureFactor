import FastTanhSinhQuadrature as FTSQ
import BSplineKit as BSK
import CurveFit as CF
using FastClosures, Printf

# Parameters

Nz = 1024+1
Nx = 200
Ny = 500
zmin = 1e-2
zmax = 1e4
z0 = 0
σ = 10

const INF = zmax
i0 = Nz÷2+1

# Initial condition profiles

function Gaussian(x, x0, σ)
    return exp(-(x-x0)^2/(2σ^2))
end

function Lorentzian(x, x0, σ)
    return 1/(1+(2(x-x0)/σ)^2)
end

# S function

function S(f, z)
    return f(z)/(z^2+f(z)^2)
end

# Landau integral

function yILandau(f, z, x, ynodes)
    Integrand = @closure y -> S(f, y)*S(f, (z*x^3+y*(1-x^2)^(3/2)))
    if z == 0 || x == 0  || x == 1 || abs(z)*(x^2/(1-x^2))^(3/2) >= INF
        I0 = FTSQ.integrate1D(Integrand, -INF, 0, ynodes...)
        I1 = 0
        I2 = FTSQ.integrate1D(Integrand, 0, INF, ynodes...)
    elseif z < 0
        I0 = FTSQ.integrate1D(Integrand, -INF, 0, ynodes...)
        I1 = FTSQ.integrate1D(Integrand, 0, -z*(x^2/(1-x^2))^(3/2), ynodes...)
        I2 = FTSQ.integrate1D(Integrand, -z*(x^2/(1-x^2))^(3/2), INF, ynodes...)
    else
        I0 = FTSQ.integrate1D(Integrand, -INF, -z*(x^2/(1-x^2))^(3/2), ynodes...)
        I1 = FTSQ.integrate1D(Integrand, -z*(x^2/(1-x^2))^(3/2), 0, ynodes...)
        I2 = FTSQ.integrate1D(Integrand, 0, INF, ynodes...)
    end
    return I0+I1+I2
end

function xILandau(f, z, xnodes, ynodes)
    Integrand = @closure x -> 2*yILandau(f, z, x, ynodes)
    return FTSQ.integrate1D(Integrand, 0, 1, xnodes...)
end

# Beliaev integral

function yIBeliaev(f, z, x, ynodes)
    Integrand = @closure y -> S(f, y)*S(f, (z-y*x^(3/2))/(1-x)^(3/2))
    if z == 0 || x == 0 || abs(z)/x^(3/2) >= INF
        I0 = FTSQ.integrate1D(Integrand, -INF, 0, ynodes...)
        I1 = 0
        I2 = FTSQ.integrate1D(Integrand, 0, INF, ynodes...)
    elseif z < 0
        I0 = FTSQ.integrate1D(Integrand, -INF, z/x^(3/2), ynodes...)
        I1 = FTSQ.integrate1D(Integrand, z/x^(3/2), 0, ynodes...)
        I2 = FTSQ.integrate1D(Integrand, 0, INF, ynodes...)
    else
        I0 = FTSQ.integrate1D(Integrand, -INF, 0, ynodes...)
        I1 = FTSQ.integrate1D(Integrand, 0, z/x^(3/2), ynodes...)
        I2 = FTSQ.integrate1D(Integrand, z/x^(3/2), INF, ynodes...)
    end
    return I0+I1+I2
end

function xIBeliaev(f, z, xnodes, ynodes)
    Integrand = @closure x -> yIBeliaev(f, z, x, ynodes)/(1-x)^(3/2)
    return 2*FTSQ.integrate1D(Integrand, 0, 1/2, xnodes...)
end

# Single iteration block

function SelfConsistentf!(f, f0, p)
    z, i0, xnodes, ynodes = p
    f_itp = BSK.interpolate(z, f0, BSK.BSplineOrder(4))
    @. f = xILandau((f_itp,), z, (xnodes,), (ynodes,)) + xIBeliaev((f_itp,), z, (xnodes,), (ynodes,))/2
    I0 = f[i0]
    println("I(z=0) = $(I0)")
    @. f = f/I0
    return nothing
end

# Fixed point iterator

function FixedPointIterator!(f!, du, u, p, tol, maxiters, tracker)
    conv = 1.0
    iters = 0
    while conv > tol
        u .= du
        f!(du, u, p)
        @. du = (du + u)/2
        conv = maximum(abs.((du .- u)./u))
        iters += 1
        if (tracker) println("Iteration $(iters) - convergence: $(conv)") end
        if (iters > maxiters) throw("Result could not converge. Try increasing tol or maxiters.") end
    end
    return nothing
end

# Self-consistent solver

function SCBAf(z::AbstractArray, f::AbstractArray; Nx::Int64=length(z), Ny::Int64=length(z), tol::Real=1e-5, maxiters::Integer=100, tracker::Bool=false)
    Nz = length(z)
    i0 = Nz÷2+1
    xnodes = FTSQ.tanhsinh(Float64, Nx)
    ynodes = FTSQ.tanhsinh(Float64, Ny)
    p = (z, i0, xnodes, ynodes)
    f0 = zeros(Nz)
    FixedPointIterator!(SelfConsistentf!, f, f0, p, tol, maxiters, tracker)
    return f
end

# Main

zlog = exp10.(range(log10(zmin), log10(zmax), length=(Nz-1)÷2)) # z > 0 array
z = cat(-reverse(zlog, dims=1), zeros(1), zlog, dims=1) # z array
f0 = Lorentzian.(z, z0, σ) # Initial condition
@time f = SCBAf(z, f0; Nx=Nx, Ny=Ny, tol=1e-5, tracker=true) # Solver call

# Data saving

open("f_Bose1D.txt", "w") do io
    println(io, "z\tf(z)")
    for i in eachindex(z)
        @printf(io, "%.18e %.18e\n", z[i], f[i])
    end
end
