M = 1e3
D = M/5
eps = 1
lam = 1

import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from numba import njit
from numba import prange
import cvxpy as cp

# --- Parametres du modele ---
T = 24*3600   # Duree totale de la simulation (24 heures en secondes)
n = 10  # Nombre de pas de temps par douche
Dt = 15 * 60  # Pas de temps pour la simulation (15 minutes en secondes)
N = int(n* 24 * 3600 / Dt)  # Nombre de groupes d'agents (toutes les 15 minutes sur 24 heures)
dT = T/N
t_vals = np.linspace(0, T, N)
n_points = 10 # Nombre de points pour les integrations numeriques

# --- Creation de la densite de probabilite des douches (rho_array) ---
def double_gaussienne(t, mu1=8*3600, mu2=20*3600, sigma=3600):
    """Double gaussienne centree a 8h et 20h pour modeliser les pics de douches."""
    return 0.5 * (np.exp(-0.5 * ((t - mu1) / sigma) ** 2) + np.exp(-0.5 * ((t - mu2) / sigma) ** 2))
rho_vals = double_gaussienne(t_vals)  # Calcul de la densite de probabilite pour chaque groupe d'agents
rho_array = rho_vals / np.sum(rho_vals)  # Normalisation pour que l'integrale soit 1

# --- Fonction de puissance moyenne ---
@njit
def P(t, puissance_moyenne=1., ecart_type=3 * 3600):
    """Puissance moyenne consommee par les agents a l'instant t."""
    return puissance_moyenne * (np.exp(-0.5 * ((t - T/2) / ecart_type) ** 2) )

# --- Calcul de C0 ---
def C_sans_renormalisation(t, f_lgn, Dt, dT, N):
    return np.sum( f_lgn* np.exp(- ((t - np.arange(N)*dT) / Dt)**2) )
integrale_P = integrate.quad(lambda t: P(t), 0, T)[0]  # Integrale totale de la puissance sur la journee
integrale_C_sans_renormalisation = integrate.quad(lambda t: C_sans_renormalisation(t, rho_array, Dt, dT, N), 0, T)[0]
C0 = integrale_P / integrale_C_sans_renormalisation

x_100 = optimize.root_scalar(lambda x: np.sinh(x) - 100, bracket=[0, 10]).root
omega0 = 2*np.pi*0.05/ x_100  # Frequence a ne pas depasser

print("Pas de discretisation :", dT, "s")
print("temps caracteristique du reseau M/D :", round(M/D, 2), "s")
print("omega0 :", round(omega0, 2), "rad/s")

# --- Fonctions principales du modele ---
@njit
def C_i(t, i, C0):
    return C0 * np.exp(- ((t - i*dT) / Dt)**2)

@njit
def C_moyen(t, f_lgn, C0, Dt, dT, N):
    return C0 * np.sum( f_lgn* np.exp(- ((t - np.arange(N)*dT) / Dt)**2) )

@njit
def omega_trapeze(t1,t2, f_lgn, C0, Dt, dT, N, n_points, D, M):
    t_grid = np.linspace(t1, t2, n_points)
    integrand = np.empty_like(t_grid)
    for idx in range(n_points):
        tt = t_grid[idx] 
        integrand[idx] = np.exp(D/M*(tt-t2)) * (P(tt) - C_moyen(tt, f_lgn, C0, Dt, dT, N))
    val_integrale = np.trapz(integrand, t_grid)
    return val_integrale / M

@njit
def omega_vect(f_lgn, C0, Dt, dT, N, n_points, D, M):
    res = np.zeros(N)
    omega_intermediaire = np.empty(N)
    for i in range(N):
        t1 = i * dT
        t2 = (i + 1) * dT
        omega_intermediaire[i] = omega_trapeze(t1, t2, f_lgn, C0, Dt, dT, N, n_points, D, M) 
        for j in range(i):
            #print(np.exp(D/M * (j+1-i)*dT))
            res[i] += omega_intermediaire[j] * np.exp(D/M * (j+1-i)*dT)
    return res

@njit(parallel=True)
def circular_slice(arr, i_min, i_max):
    N = len(arr)
    start = i_min % N
    end = i_max % N
    if start <= end:
        return arr[start:end + 1]
    else:
        size = N - start + end + 1
        out = np.empty(size, arr.dtype)
        for k in prange(N - start):
            out[k] = arr[start + k]
        for k in prange(end + 1):
            out[N - start + k] = arr[k]
    return out

@njit(parallel=True)
def Couts_integrales(f_lgn, Dt, dT, N, n_points, D, M, eps, omega_array, omega0):
    res = np.empty(N)
    for i in prange(N):
        omega_slice = circular_slice(omega_array, i - 3*n, i + 3*n)
        t_slice = circular_slice(np.arange(0, N) * dT, i - 3*n, i + 3*n)
        integrand = np.empty_like(omega_slice)
        for j in range(len(omega_slice)):
            integrand[j] = np.sinh(omega_slice[j]/omega0) * C_i(t_slice[j], i, C0) #np.sinh(omega_slice[j]/omega0)
        res[i] = eps * np.trapz(integrand, np.arange(0, 6*n+1 )*dT)
    return res

@njit(parallel=True)
def Couts_totaux(j, Couts_integral):
    res = np.empty(N)
    for i in prange(N):
        res[i] = lam * (i - j) ** 2 - Couts_integral[i]
    return res

def affichage(f_lgn, omega_array, commentaire = ""):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(t_vals / 3600, [C_moyen(t, f_lgn, C0, Dt, dT, N) for t in t_vals], label="Consommation d'energie")
    axs[0].plot(t_vals / 3600, [C_moyen(t, rho_array, C0, Dt, dT, N) for t in t_vals], label="Consommation sans opti")
    axs[0].plot(t_vals / 3600, [P(t) for t in t_vals], label='Puissance moyenne')
    axs[0].set_xlabel('Heure de la journee', fontsize=15)
    axs[0].set_ylabel('Puissance (W)', fontsize=15)
    axs[0].set_title('Consommation et puissance moyennes {}'.format(commentaire), fontsize=15)
    axs[0].legend()
    axs[1].plot(t_vals / 3600, omega_array, label="Omega")
    axs[1].set_xlabel('Heure de la journee', fontsize=15)
    axs[1].set_ylabel('Omega (rad/s)', fontsize=15)
    axs[1].set_title('Omega moyen {}'.format(commentaire), fontsize=15)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def plot_strategie(f_array):
    max_vals = f_array.max(axis=0)
    f_flatten = f_array.flatten().copy()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(max_vals, label='Strategie maximale')
    axs[0].set_xlabel("Indice de colonne", fontsize=14)
    axs[0].set_ylabel("Valeur maximale", fontsize=14)
    axs[0].set_title("Maximum par colonne", fontsize=16)
    axs[0].legend()
    axs[1].hist(f_flatten[f_flatten > 1e-1], bins=100, color='skyblue', edgecolor='black')
    axs[1].set_xlabel("Valeur (> 0.1)", fontsize=14)
    axs[1].set_ylabel("Nombre d'occurrences", fontsize=14)
    axs[1].set_title("Histogramme des valeurs de la strategie", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- Recherche de l'equilibre de Nash par point fixe avec ralentissement ---
f_array = np.random.dirichlet(np.ones(N), size=N).T
alphas = np.array([0.1, 0.01, 0.01])
nb_iters = np.array([30, 50, 100])

f_lgn = np.dot(f_array, rho_array)  # Distribution des agents actifs
omega_lgn = np.cumsum(np.array([omega_trapeze(i*dT, (i+1)*dT, f_lgn, C0, Dt, dT, N, n_points, D, M) for i in range(N)]))
affichage(f_lgn, omega_lgn, commentaire = "avant optimisation")
plot_strategie(f_array)

for k in range(len(nb_iters)):
    print(f"Phase {k+1}/{len(nb_iters)}: {nb_iters[k]} iterations avec alpha = {alphas[k]}")
    nb_iter = nb_iters[k]
    alpha = alphas[k]
    for it in tqdm(range(nb_iter)):
        f_lgn = np.dot(f_array, rho_array)
        omega_lgn = omega_vect(f_lgn, C0, Dt, dT, N, n_points, D, M)
        Couts_integral_array = Couts_integrales(f_lgn, Dt, dT, N, n_points, D, M, eps, omega_lgn, omega0)

        for j in range(N):
            Couts_total_array = Couts_totaux(j, Couts_integral_array)
            i_etoile = np.argmin(Couts_total_array)  # Strategie optimale pour les agents du groupe j
            f_array[:,j] = (1- alpha) *f_array[:,j] + np.array([alpha if i == i_etoile else 0 for i in range(N)])
    affichage(f_lgn, omega_lgn)
    plot_strategie(f_array)
