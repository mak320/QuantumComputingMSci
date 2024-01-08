""""
Tesing the impleentation of the F1 map with coupling to gauss's law
Jan 2 2024
Mate 
"""


import numpy as np
import matplotlib.pyplot as plt
from math import ceil

FontSize = 20
# define paramters
Nv = 10  # number of velocity points
Nx = 10  # number of space points
N = Nv*Nx

dv = 1  # velocity step
dx = 1  # space step

xmax = (Nx-1) * dx / 2  # maximum space (not independent)
vmax = (Nv-1) * dv / 2  # maximum velocity (not independent)

q = 1  # charge
m= 1 # mass 
eps_0 = 10  # permittivity
N_particles = 1



def ddslash(a, b):  # a/b with integer division
    return a - b * (ceil(a/b) - 1)


def kron_delta(a, b):  # § Kronecker delta
    a = int(a)
    b = int(b)
    if a == b:
        return 1
    else:
        return 0
    
def vj(j):
    vmax = (Nv-1) * dv /2 
    return -vmax + (j-1) * dv

def create_F1_hat():
    F1_regular = np.zeros((N,N))
    F1_gauss = np.zeros((N,N))

    for n in range(0, N):
        n_phys = n + 1
        v_nddNv = vj(ddslash(n_phys, Nv))
        prefact = -v_nddNv / (2 * dx)

        if n_phys <= Nv:
            for k1 in range(0, N):
                k1_phys = k1 + 1
                F1_regular[n, k1] = prefact * (kron_delta(k1_phys, n_phys + Nv) -
                                                kron_delta(k1_phys, n_phys + Nv * (Nx-1))) 
            
        elif Nv < n_phys <= Nv*(Nx-1):
            for k2 in range(0, N):
                k2_phys = k2 + 1
                F1_regular[n, k2] = prefact * (kron_delta(k2_phys, n_phys + Nv) -
                                                kron_delta(k2_phys, n_phys - Nv))
        
        else:
            for k3 in range(0, N):
                k3_phys = k3 + 1
                F1_regular[n, k3] = prefact * (kron_delta(k3_phys, n_phys - Nv * (Nx -1)) -
                                                kron_delta(k3_phys, n_phys - Nv))
                
    for n in range(0, N):
        n_phys = n + 1
        prefact_gauss = (-q**2 * N_particles) / (2 * m * eps_0 * dv) * (ceil(n_phys/Nv) - 1) / (Nx - 1)
        
        if ddslash(n_phys, Nv) == 1:
            for k1 in range(0, N):
                k1_phys = k1 + 1
                F1_gauss[n, k1] = prefact_gauss * kron_delta(k1_phys, n_phys +1)
        elif ddslash(n_phys, Nv) == Nv:
            for k2 in range(0, N):
                k2_phys = k2 + 1
                F1_gauss[n, k2] = -prefact_gauss * kron_delta(k2_phys, n_phys -1)
        else: 
            for k3 in range(0, N):
                k3_phys = k3 + 1
                F1_gauss[n, k3] = prefact_gauss * (kron_delta(k3_phys, n_phys +1) - 
                                                kron_delta(k3_phys, n_phys -1))  

    return F1_regular + F1_gauss


F1_hat_gauss = create_F1_hat()

plt.imshow(F1_hat_gauss, cmap="plasma")
plt.xlabel("n", fontsize=FontSize)
plt.ylabel("k", fontsize=FontSize)
plt.title(r"$[\hat{F}^{(1)}]_{n, k}$", fontsize=FontSize)
plt.colorbar()
plt.show()


f_init = np.random.rand(Nx, Nv)
dt = 1
# f = np.ones((Nx, Nv))

f_init = np.zeros((Nx, Nv))
for i in range(0, Nx):
    for j in range(0, Nv):
        if i % 2 == 0: # ha a sor vagy oszlop paros
            f_init[i, j] = 1
        else:
            f_init[i, j] = 2

f_init = f_init * N_particles / (np.sum(f_init) * (Nx-1) * dx * (Nv-1) * dv)

plt.title("Initial distribution")

plt.imshow(f_init, cmap="plasma")
plt.colorbar()
plt.show()

def u_n(f):
    un = np.zeros(N)
    for n in range(0, N):
        n_phys = n +1
        un[n] = f[ceil(n_phys/Nv) -1, ddslash(n_phys, Nv) -1] 
    return un

u_init = u_n(f_init)


u_matrix_ev = u_init + dt * F1_hat_gauss @ u_init

def dfdt(f):
    # Define arrays for the derivatives
    dfdx_term = np.zeros((Nx, Nv))
    dfdv_term = np.zeros((Nx, Nv))
    
    # Perform one time step of iteration using a second-order finite difference scheme
    for i in range(0, Nx):
        i_phys = i + 1
        prefact_gauss = (-q**2 * N_particles) / (2 * m * eps_0 * dv) * (i_phys - 1) / (Nx - 1)
        for j in range(0, Nv):
            j_phys = j + 1
            vj = (-vmax + (j_phys - 1) * dv)
            # implement spacial derivative with peridic boudary contions
            if i_phys == 1: # i = 0
                # f_{2, j} - f_{N_x, j} # shift all physical indicies back by 1
                dfdx_term[i, j] = -vj * (f[1, j] - f[Nx-1, j]) / (2 * dx)
            elif i_phys == Nx: # i = Nx - 1
                dfdx_term[i, j] = -vj * (f[0, j] - f[Nx-2, j]) / (2 * dx)
            else:
                dfdx_term[i, j] = -vj * (f[i+1, j] - f[i-1, j]) / (2 * dx)
            

            # implementing velocity derivative with fixed boundary conditions
            if j_phys == 1: # j = 0
                # f_{i, 2} - 0 
                dfdv_term[i, j] = prefact_gauss * (f[i, 1]) / (2 * dv)

            if j_phys == Nv:
                # 0 - f_{i, N_v-1} 
                dfdv_term[i, j] = prefact_gauss * (-f[i, Nv-2]) / (2 * dv) ## egyesevel vegig neztem 
            else:
                dfdv_term[i, j] = prefact_gauss * (f[i, j+1] - f[i, j-1]) / (2 * dv)
            
    dfdt = dfdx_term + dfdv_term

    return dfdt

dfdt = dfdt(f_init)



f_direct_ev = f_init + dt * dfdt

u_direct_ev = u_n(f_direct_ev)


diff = (u_matrix_ev - u_direct_ev) / (u_direct_ev)

x_axis = np.arange(0, Nx*Nv, 1)

fig = plt.figure()
ax1 = fig.add_subplot(211)


ax1.plot(x_axis, u_matrix_ev, 'o', color='black', label='F1', alpha=0.6, ms=7)
ax1.plot(x_axis, u_direct_ev, 'o', color='red',label='direct', ms=3, alpha=0.9)
# ax1.plot(x_axis,inintial_un,'o',color='black',label='initial', alpha=0.3, ms = 15)


ax1.set_ylabel(r"$u_n$", fontsize=FontSize)
ax1.legend(fontsize=FontSize)



ax2 = fig.add_subplot(212)


plt.plot(diff, "ro--")
ax2.set_xlabel(r"$n$", fontsize=FontSize)
ax2.set_ylabel(r"$\frac{u_n^{F1} - u_n^{direct}}{u_n^{0}}$", fontsize=FontSize)

plt.tight_layout()
plt.show()