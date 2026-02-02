using DifferentialEquations, Plots, LinearAlgebra, Distributions, ProgressMeter, NPZ, LaTeXStrings, Measures

# Parametres
K = 8.0
tspan = (0.0, 1000.0)
ϵ = 1e-3
Omega0 = 1e-1

n = 100
Alpha = range(1e-1, 6, length=n)
P_liste = range(1e-1, 10, length=n)
m = length(P_liste)

results = Array{Float64, 3}(undef, n, m, 2)
colors = Matrix{Symbol}(undef, n, m)

@showprogress for i in 1:n
    α = Alpha[i]
    for (j, P) in enumerate(P_liste)
        if P>K
            colors[i,j] = :grey
        else
            u0 = [π - asin(P / K), Omega0]
            p = (P, α, K)
            function f_ode(du, u, p, t)
                P, α, K = p
                θ, ω = u
                du[1] = ω
                du[2] = 2P - α * ω - 2K * sin(θ)
            end
            prob = ODEProblem(f_ode, u0, tspan, p)
            sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
            colors[i,j] = abs( sol.u[end][2] / (2π) ) < ϵ ? :red : :yellow
        end
        results[i, j, 1] = α
        results[i, j, 2] = P
    end
end

# --- Affichage ---
gr()
s = scatter(
    results[:,:,1],
    results[:,:,2],
    color=colors[:,:],
    markersize = 2,
    markerstrokecolor = colors[:,:],
    legend=false,
    label = ""
    )
scatter!(s, [NaN], [NaN], color=:red, markerstrokecolor=:red, label="Point fixe globalement stable")
scatter!(s, [NaN], [NaN], color=:grey, markerstrokecolor=:grey, label="Cycle attracteur globalement stable")
scatter!(s, [NaN], [NaN], color=:yellow, markerstrokecolor=:yellow, label="Point fixe et cycle attracteur localement stables")
plt = plot(s,
    size=(900,600),
    legend = :bottomright,
    title=L"K = 8.0~[\mathrm{rad.s^{-2}}]",
    xlabel=L"\alpha~\textit{(dumping)}~[\mathrm{s^{-1}}]",
    ylabel=L"P~[\mathrm{rad.s^{-2}}]",
    xlabelfontsize=22,
    ylabelfontsize=22,
    titlefontsize=22,
    tickfontsize=18,
    legendfontsize=12,
    left_margin=8mm,
    bottom_margin=5mm,
    top_margin=5mm, 
    )
savefig(plt, "diagramme_de_bifurcation_par.pdf")
display(plt)
