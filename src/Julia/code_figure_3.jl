using DifferentialEquations, Plots, LinearAlgebra, Distributions, ProgressMeter, LaTeXStrings, Measures
gr()

# Parametres
K = 1.0
α = 0.3
γ = 0.0
τ = 0.0
tspan = (0.0, 600.0)
ϵ = 2π * 1e-1
P_liste = [0.4, 0.6]
dephasage = 0.0

# --- Calcul des points bleus et verts ---
n = 20000
m = length(P_liste)
zeta_theta = π
zeta_omega = 30.0
dTheta = rand(Uniform(-zeta_theta, zeta_theta), n)
dOmega = rand(Uniform(-zeta_omega, zeta_omega), n)

results = Array{Float64, 3}(undef, n, m, 2)
colors = Matrix{Symbol}(undef, n, m)

@showprogress for i in 1:n
    u0 = [dTheta[i], dOmega[i]]
    for (j, P) in enumerate(P_liste)
        p = (P, α, K, γ, τ)
        if τ == 0.0
            function f_ode(du, u, p, t)
                P, α, K, γ, τ = p
                θ, ω = u
                du[1] = ω
                du[2] = 2P - (α + γ) * ω - 2K * sin(θ + dephasage)
            end
            prob = ODEProblem(f_ode, u0, tspan, p)
            sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=0.01)
        else
            function h(p, t)
                scale = 1.0 + 0.1 * tanh(t / 2)
                return [u0[1] * scale, u0[2] * scale]
            end
            function f_dde(du, u, h, p, t)
                P, α, K, γ, τ = p
                θ, ω = u
                ωτ = h(p, t - τ)[2]
                du[1] = ω
                du[2] = 2P - α * ω - 2K * sin(θ + dephasage) - γ * ωτ
            end
            prob = DDEProblem(f_dde, u0, h, tspan, p; constant_lags=[τ])
            sol = solve(prob, reltol=1e-8, abstol=1e-8, saveat=0.01)
        end
        dθi = sol.u[1][1]
        dωi, dωf = sol.u[1][2], sol.u[end][2]
        results[i, j, :] = [dθi, dωi]
        colors[i,j] = abs(dωf) < ϵ ? :green : :blue

    end
end

# --- Calcul du cycle attracteur ---
P_liste = [0.4, 0.6]
trajectoire = Array{Float64, 3}(undef, 2, 2, 300)
u0= [0.0, 15.0]
tspan = (0.0, 600.0)
for (j, P) in enumerate(P_liste)
    p = (P, α, K, γ, τ)
    function f_ode(du, u, p, t)
        P, α, K, γ, τ = p
        θ, ω = u
        du[1] = ω
        du[2] = 2P - (α + γ) * ω - 2K * sin(θ + dephasage)
    end
    prob = ODEProblem(f_ode, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=0.01)
    θs = [u[1] for u in sol.u[end-299:end]]
    ωs = [u[2] for u in sol.u[end-299:end]]
    trajectoire[j, 1, :] = θs
    trajectoire[j, 2, :] = ωs
end

function wrap_to_pi(arr)
    return [mod(x + π, 2π) - π for x in arr]
end
trajectoire[:,1,:] = wrap_to_pi(trajectoire[:,1,:])

for i in axes(trajectoire, 1)
    for j in axes(trajectoire, 2)
        # Recupere les indices de tri selon la valeur de la 2e coordonnee
        idx = sortperm(trajectoire[i, 1, :])
        # Trie les deux coordonnees selon ces indices
        trajectoire[i, 1, :] = trajectoire[i, 1, idx]
        trajectoire[i, 2, :] = trajectoire[i, 2, idx]
    end
end

# --- affichage des points bleus et verts ---
p1 = scatter(results[:,1,1], results[:,1,2], title=L"P = 0.4~[\mathrm{rad.s^{-2}}]", color=colors[:,1], markersize = 2, markerstrokecolor = colors[:,1], legend=false,label = "")
p2 = scatter(results[:,2,1], results[:,2,2], title=L"P = 0.6~[\mathrm{rad.s^{-2}}]", color=colors[:,2], markersize = 2, markerstrokecolor = colors[:,2], legend=false, label = "")
scatter!(p1, [NaN], [NaN], color=:green, markerstrokecolor=:green, label="Converge vers le point fixe")
scatter!(p1, [NaN], [NaN], color=:blue, markerstrokecolor=:blue, label="Atteint le cycle limite")

# --- affichage du cycle attracteur ---
plot!(p1, trajectoire[1, 1, :], trajectoire[1, 2, :], color=:red, linewidth=3, label="Attracteur")
plot!(p2, trajectoire[2, 1, :], trajectoire[2, 2, :], color=:red, linewidth=3, label="")

# --- affichage du point fixe ---
scatter!(p1, [asin(P_liste[1]/K)], [0], color=:red, markerstrokecolor=:red, marker=:star6, markersize=12, label="Point fixe stable")
scatter!(p2, [asin(P_liste[2]/K)], [0], color=:red, markerstrokecolor=:red, marker=:star6, markersize=12, label="")

# --- generation et enregistrement de l'image ---
plt = plot(p1, p2,
    layout=(1,2),
    size=(1200,600),
    legend=:topleft,
    xlabel=L"\theta_i~[\mathrm{rad}]", 
    ylabel=L"\omega_i~[\mathrm{rad.{s}^{-1}}]", 
    xlabelfontsize=22,
    ylabelfontsize=22,
    titlefontsize=22,
    tickfontsize=18,
    legendfontsize=12,
    left_margin=8mm,
    bottom_margin=15mm,
    top_margin=10mm,  
    xticks=([-π,-π/2, 0, π/2, π], [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]),
    yticks=([-30, 0, 30], [L"-30", L"0", L"30"]),
    framestyle=:box,
    framestyle_color=:black,
    framestyle_width=1.5,
    grid=:on,
)
display(plt)
savefig(plt, "diagramme_bifurcation.pdf")
