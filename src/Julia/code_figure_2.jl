using DifferentialEquations, Plots, LinearAlgebra, Distributions, ProgressMeter, NPZ, LaTeXStrings, Measures
gr()

# --- Calcul de l'energie potentielle ---
Theta = range(-π, 3π, length=1000)
K = 1.0
P_liste = [0.1, 0.4, 0.6, 1.1]
E_val = Array{Float64}(undef, length(Theta), length(P_liste))
Theta_max = zeros(length(P_liste))
E_max = zeros(length(P_liste))
for (j, P) in enumerate(P_liste)
    function E(θ)
        return - 2* P * θ - 2* K * cos(θ)
    end
    E_val[:,j] = E.(Theta)
    if P < K
        Theta_max[j] = π - asin(P/K)
        E_max[j] = E(Theta_max[j])
    end
end

# --- Trace de l'energie potentielle ---
p1 = plot(Theta, E_val[:,1], title=L"P = 0.1~[\mathrm{rad.s^{-2}}]", color= [0 < theta < 2π ? :blue : :grey for theta in Theta], linewidth=2, label="V(θ)")
p2 = plot(Theta, E_val[:,2], title=L"P = 0.4~[\mathrm{rad.s^{-2}}]", color= [0 < theta < 2π ? :blue : :grey for theta in Theta], linewidth=2, label="V(θ)")
p3 = plot(Theta, E_val[:,3], title=L"P = 0.6~[\mathrm{rad.s^{-2}}]", color= [0 < theta < 2π ? :blue : :grey for theta in Theta], linewidth=2, label="V(θ)")
p4 = plot(Theta, E_val[:,4], title=L"P = 1.1~[\mathrm{rad.s^{-2}}]", color= [0 < theta < 2π ? :blue : :grey for theta in Theta], linewidth=2, label="V(θ)")
p1 = scatter!(p1, [Theta_max[1]], [E_max[1]], color=:black, linestyle=:dash, label="Max local de V")
p2 = scatter!(p2, [Theta_max[2]], [E_max[2]], color=:black, linestyle=:dash, label="Max local de V")
p3 = scatter!(p3, [Theta_max[3]], [E_max[3]], color=:black, linestyle=:dash, label="Max local de V")
plt = plot(p1, p4, p3, p2, 
    layout=(2,2), 
    size=(1200,900), 
    xlabel=L"\theta_i~[\mathrm{rad}]", 
    ylabel=L"V(\theta_i)",
    xticks=([-π, 0, π, 2π, 3π], [L"-\pi", L"0", L"\pi", L"2\pi", L"3\pi"]),
    legend=:topright,
    xlabelfontsize=22,   # Taille de la legende de l'axe x
    ylabelfontsize=22,    # Taille de la legende de l'axe y
    titlefontsize=22,     # Taille du titre
    tickfontsize=18,      # Taille des ticks
    legendfontsize=12,     # Taille de la legende
    left_margin=10mm, 
    bottom_margin=10mm
)
savefig(plt, "energie_potentielle.pdf")
display(plt)