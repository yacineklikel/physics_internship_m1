#  ----  variables physiques  ----
M = 1e3
P0 = 1e-1
lam = 1
kappa = 1e0
D = 1e-1

#  ----  initialisation  ----
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
plt.close('all')

T = 3600
N_t = 1000
dt = T / N_t
N_theta, N_omega = 1000, 1000

print("Pas de discretisation :", dt, "s")
omega_0 = 2*np.pi* 0.05
print("omega_0 :", omega_0, "rad/s")
theta_grid, omega_grid = np.meshgrid(
    np.linspace(-np.pi, np.pi, N_theta, endpoint=False),
    np.linspace(-2/3*omega_0, 2/3*omega_0, N_omega, endpoint=False),
    indexing='ij'
)
dtheta = (theta_grid[-1, 0] - theta_grid[0, 0]) /(N_theta - 1)
domega = (omega_grid[0, -1] - omega_grid[0, 0]) /(N_omega - 1)

# ----  fonctions du code  ----
@njit
def P(t):
    return P0*np.ones((N_theta, N_omega), dtype=np.float64)

@njit(parallel=True)
def demi_evolution_theta(arr):
    new_arr = np.empty_like(arr)
    for i in prange(N_theta):
        for j in range(N_omega):
            i_deplacement = omega_grid[0,j]*((dt/2)/dtheta)
            if i_deplacement >= 0:
                new_arr[i, j] = (1 - i_deplacement%1) * arr[(i - int(i_deplacement))%(N_theta-1), j] + \
                                i_deplacement%1 * arr[(i - int(i_deplacement+1))%(N_theta-1), j]          # la masse qui arrive en i cest celle qui etait en i-v*dt (dt/2 car demi-evolution pour respecter Stang)
            else:
                i_deplacement = -i_deplacement
                new_arr[i, j] =i_deplacement%1 * arr[(i + int(i_deplacement+1))%(N_theta-1), j] + \
                                (1 - i_deplacement%1) * arr[(i + int(i_deplacement))%(N_theta-1), j]
    return new_arr

@njit(parallel=True)
def d_domega(arr):
    grad_omega = np.empty_like(arr)
    for i in prange(N_theta):
        for j in range(N_omega):
            if 0 < j < N_omega-1:
                grad_omega[i, j] = (arr[i, j+1] - arr[i, j-1]) / (2*domega)
            elif j == 0:
                grad_omega[i, j] = (arr[i, j+1] - arr[i, j]) / domega
            else:
                grad_omega[i, j] = (arr[i, j] - arr[i, j-1]) / domega
    return grad_omega

@njit(parallel=True)
def drho_dt(rho_t, Pt, r, Psi):
    return -1/M * d_domega( (-kappa*omega_grid + Pt + lam*r*np.sin(Psi - theta_grid)) * rho_t) + \
           2*(D/M)**2 * (d_domega(d_domega(rho_t)) )

@njit(parallel=True)
def runge_kutta_4(i_t, rho_t, r, Psi):
    k1 = drho_dt(rho_t,            P(i_t//N_t),                        r, Psi)
    k2 = drho_dt(rho_t + k1*dt/2, (P(i_t//N_t) + P((i_t+1)//N_t) ) /2, r, Psi)  # Pour evaluer P en t + dt/2, on prend la moyenne des P(t) et P(t+dt/2)
    k3 = drho_dt(rho_t + k2*dt/2, (P(i_t//N_t) + P((i_t+1)//N_t) ) /2, r, Psi)
    k4 = drho_dt(rho_t + k3*dt,    P((i_t+1)//N_t),                    r, Psi)  # Ici on evalue P en t + dt, donc on prend P(t+dt)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

#  ----  boucle et affichage  ----
rho_t = np.exp(-(20*(omega_grid)/omega_0)**2 / 2)*np.exp(-(10*theta_grid/np.pi)**2 / 2)
rho_t /= np.sum(rho_t)
rPsi_list = []
for i_t in tqdm(range(N_t)):
    rPsi_list.append(  np.mean(np.exp(1j*theta_grid[:,:])*rho_t[:,:]) )
    r = np.abs(rPsi_list[-1])
    Psi = np.angle(rPsi_list[-1])
    rho_t = demi_evolution_theta(rho_t)             # Deplacement explicite suivant theta
    rho_t += runge_kutta_4(i_t, rho_t, r, Psi)      # RK4 pour l'evolution en omega
    rho_t[:, 0] = 0.0
    rho_t[:, -1] = 0.0
    rho_t = np.clip(rho_t, 0, None)                 
    rho_t /= np.sum(rho_t)                       
    rho_t = demi_evolution_theta(rho_t)             # Deplacement explicite suivant theta
    # Inutile en theorie mais evite erreurs d'arrondis
    rho_t[:, 0] = 0.0
    rho_t[:, -1] = 0.0
    rho_t[rho_t < 0] = 0
    rho_t /= np.sum(rho_t)
    if i_t % (N_t // 10) == 0:
        plt.figure(figsize=(3,3))
        plt.pcolormesh(omega_grid, theta_grid, rho_t, shading='auto', cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel('Omega (rad/s)')
        plt.ylabel('Theta (rad)')
        plt.title(f'Temps: {round(i_t*dt/60, 1)} min')
        plt.show()
plt.figure(figsize=(3,3))
plt.pcolormesh(omega_grid, theta_grid, rho_t, shading='auto', cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('Omega (rad/s)')
plt.ylabel('Theta (rad)')
plt.title(f'Temps: {round(i_t*dt/60, 1)} min')
plt.show()