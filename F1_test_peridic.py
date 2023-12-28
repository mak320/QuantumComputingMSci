"""
Testing script for the F1 discretization
Nov 11 2023

"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

FontSize = 20
# define paramters
Nv = 10  # number of velocity points
Nx = 10  # number of space points

dv = 1  # velocity step
dx = 1  # space step

vmax = (Nv-1) * dv / 2  # maximum velocity (not independent)

q = 1  # charge
eps_0 = 1  # permittivity
c = dv * q / eps_0


def ddslash(a, b):  # a/b with integer division
    return a - b * (ceil(a/b) - 1)


def kron_delta(a, b):  # § Kronecker delta
    a = int(a)
    b = int(b)
    if a == b:
        return 1
    else:
        return 0


# converts the f_ij and E_i arrays to a single array of U_n values
def U_n_converter(f_arr, E_arr):
    un = np.zeros(Nx*(Nv+1))
    for n in range(0, Nx*(Nv + 1)):
        n_phys = n + 1
        if n_phys <= Nx * Nv:
            un[n] = f_arr[ceil(n_phys/Nv) - 1, ddslash(n_phys, Nv) - 1]
        else:
            un[n] = E_arr[n_phys - Nx*Nv - 1]
    return un


def create_F1_matrix():
    F1 = np.zeros(((Nv+1)*Nx, (Nv+1)*Nx))
    N = (Nv+1)*Nx

    for n in range(0, (Nv+1)*Nx):
        n_phys = n + 1
        v_nddNv = -vmax + (ddslash(n_phys, Nv) - 1) * dv

        pre_fact = - v_nddNv / (2 * dx)

        if n_phys <= Nv:
            for k1 in range(0, N):
                k1_phys = k1 + 1
                F1[n, k1] = pre_fact * (kron_delta(k1_phys, n_phys + Nv) +
                                        kron_delta(k1_phys, n_phys + Nv * (Nx-1)))

        elif n_phys > Nv and n_phys <= Nv*(Nx-1):
            for k2 in range(0, N):
                k2_phys = k2 + 1
                F1[n, k2] = pre_fact * \
                    (kron_delta(k2_phys, n_phys + Nv) -
                     kron_delta(k2_phys, n_phys - Nv))

        elif n_phys > Nv*(Nx-1) and n_phys <= Nv*Nx:
            for k3 in range(0, N):
                k3_phys = k3 + 1
                F1[n, k3] = pre_fact * (kron_delta(k3_phys, n_phys -
                                        Nv * (Nx-1)) + kron_delta(k3_phys, n_phys - Nv))

        else:
            for k in range(0, (Nv+1)*Nx):
                k_phys = k + 1
                S = 0
                for j in range(1, Nv+1):  # here j is a physical index
                    v_j = -vmax + (j - 1) * dv
                    # S += kron_delta(k_phys, (ddslash(n_phys, Nv) - 1) * Nv + j) * (-vmax + (j - 1) * dv)
                    S += kron_delta(k_phys, (n_phys - Nx *
                                    Nv - 1) * Nv + j) * v_j

                # boundary = (vmax/2)  * (kron_delta(k_phys, (ddslash(n_phys, Nv) - 1) * Nv + 1) - kron_delta(k_phys, ddslash(n_phys, Nv)))
                boundary = (vmax/2) * (kron_delta(k_phys, (n_phys - Nx * Nv - 1)
                                                  * Nv + 1) - kron_delta(k_phys, (n_phys - Nx * Nv - 1)*Nv + Nv))

                F1[n, k] = c * (S + boundary)
    return F1


F1 = create_F1_matrix()

plt.imshow(F1, cmap="plasma")
plt.xlabel("n", fontsize=FontSize)
plt.ylabel("k", fontsize=FontSize)
plt.title(r"$[\hat{F}^{(1)}]_{n, k}$", fontsize=FontSize)
plt.colorbar()
plt.show()
np.random.seed(111)

f = np.random.rand(Nx, Nv)
f = f / (np.sum(f) * (Nx-1) * dx * (Nv-1) * dv)

# sums out the rows of the f_ij matrix, the rows corsopnd to constant space slices of f_ij, so this is exactly what we need to approimate an integral over velocities
int_f_dv = np.sum(f, axis=1)

E_i = np.cumsum(int_f_dv)

dt = 1

inintial_un = U_n_converter(f, E_i)

result_F1 = F1 @ inintial_un * dt + inintial_un


def progate_u():

    def propogate_f(i, j):  # i and j are physical indicei
        vj = (vmax - (j - 1) * dv) / dx
        ii = i - 1
        ji = j - 1
        if i == 1:
            return vj * (-3/2 * f[ii, ji] + 2 * f[ii+1, ji] - 1/2 * f[ii+2, ji])
        elif i == Nx:
            return vj * (3/2 * f[ii, ji] - 2 * f[ii-1, ji] + 1/2 * f[ii-2, ji])
        else:
            return vj * (-1/2 * f[ii-1, ji] + 1/2 * f[ii+1, ji])

    evolved_f = np.zeros((Nx, Nv))
    evolved_E_i = np.zeros(Nx)
    for i in range(0, Nx):
        i_phys = i + 1
        S = 0
        boundary = (vmax/2) * (f[i, 0] - f[i, Nv-1])
        for j in range(0, Nv):
            j_phys = j + 1

            S += (f[i, j] * (-vmax + (j_phys-1) * dv))

            evolved_f[i, j] = f[i, j] + propogate_f(i_phys, j_phys) * dt

        evolved_E_i[i] = E_i[i] + c * (S + boundary) * dt

    return U_n_converter(evolved_f, evolved_E_i)


def propogate_u_no_conversion():
    """
    implemented equation 5.16 directly form latex
    """
    evolved_un = np.zeros(Nx*(Nv+1))
    for n in range(0, Nx*(Nv+1)):
        n_phys = n + 1
        pre_fact = (vmax - (ddslash(n_phys, Nv) - 1) * dv) / (dx)

        if n_phys <= Nv:
            du_dt = pre_fact * \
                (-3/2 * inintial_un[n] + 2 * inintial_un[n +
                 Nv] - 1/2 * inintial_un[n + 2*Nv])

        elif Nv < n_phys and n_phys <= Nv*(Nx-1):
            du_dt = pre_fact * \
                (1/2 * inintial_un[n + Nv] - 1/2 * inintial_un[n - Nv])

        elif n_phys > Nv*(Nx-1) and n_phys <= Nv*Nx:
            du_dt = pre_fact * \
                (3/2 * inintial_un[n] - 2 * inintial_un[n -
                 Nv] + 1/2 * inintial_un[n - 2*Nv])

        else:
            S = 0
            boundary = vmax/2 * (inintial_un[(ddslash(n_phys, max(Nx, Nv))-1)
                                 * Nv + 1 - 1] - inintial_un[ddslash(n_phys, max(Nx, Nv))-1])
            for j in range(1, Nv+1):
                S += inintial_un[(ddslash(n_phys, max(Nx, Nv))-1)
                                 * Nv + j - 1] * (-vmax + (j-1)*dv)

            du_dt = c * (S + boundary)

        evolved_un[n] = inintial_un[n] + du_dt * dt

    return evolved_un


result_direct = progate_u()

(result_F1)
(result_direct)
x_axis = np.arange(0, Nx*(Nv+1), 1)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.vlines(Nv, np.min(inintial_un), np.max(
    inintial_un), color="black", linestyles="dashed")
ax1.vlines(Nv*(Nx-1), np.min(inintial_un), np.max(inintial_un),
           color="black", linestyles="dashed")
ax1.vlines(Nv*Nx, np.min(inintial_un), np.max(inintial_un),
           color="black", linestyles="dashed")

ax1.plot(x_axis, result_F1, 'o', color='black', label='F1', alpha=0.6, ms=7)
ax1.plot(x_axis, result_direct, 'o', color='red',
         label='direct', ms=3, alpha=0.9)
# ax1.plot(x_axis,inintial_un,'o',color='black',label='initial', alpha=0.3, ms = 15)


ax1.set_ylabel(r"$u_n$", fontsize=FontSize)
ax1.legend(fontsize=FontSize)


# S = 0
# i = 1
# boudnary = vmax/2 * (f[i-1, 0] - f[i-1, Nv-1])
# for j in range(0, Nv):
#     j_phys = j + 1
#     S += (f[i-1, j] * (-vmax + (j_phys-1) *dv))

# dE_dt  = c * (S + boudnary)

# evolved_E = E_i[i-1] + dE_dt * dt

# plt.plot(Nx*Nv+i-1, evolved_E, "*", color="orange")


"""testing the 1st segment"""
# m=5
# du_dt = (vmax - (ddslash(m,Nv) -1)*dv) * (-3/2 * inintial_un[m-1] + 2 * inintial_un[m-1+Nv] - 1/2 * inintial_un[m-1 + 2*Nv])
# evolved_un = inintial_un[m-1] + du_dt * dt
# plt.plot(m-1, evolved_un, "*", color="orange")


"""testing the 2nd segment"""
# m = 45
# du_dt = (vmax - (ddslash(m,Nv) -1)*dv) * (1/2 * inintial_un[m + Nv -1] - 1/2 * inintial_un[m-1 - Nv])
# evolved_un = inintial_un[m-1] + du_dt * dt
# plt.plot(m-1, evolved_un, "*", color="orange")


"""testing the 3rd segment"""
# m = 50
# du_dt = (vmax - (ddslash(m,Nv) -1)*dv) * (3/2 * inintial_un[m-1] - 2 * inintial_un[m-1 - Nv] + 1/2 * inintial_un[m-1 - 2*Nv])
# evolved_un = inintial_un[m-1] + du_dt * dt
# plt.plot(m-1, evolved_un, "*", color="orange")


"""testing the 4th segment (electric field)"""
# m = Nx * Nv + i
# S = 0
# boundary = vmax/2 * (inintial_un[(ddslash(m, Nv)-1)*Nv + 1 - 1] - inintial_un[ddslash(m, Nv)-1])
# for j in range(1, Nv+1):
#     S +=  inintial_un[(ddslash(m, Nv)-1)*Nv + j - 1] * (-vmax + (j-1)*dv)

# du_dt = c * (S + boundary)
# evolved_un = inintial_un[m-1] + du_dt * dt
# plt.plot(m-1, evolved_un, "*", color="red", ms = 10, alpha=0.5)

error = np.abs(result_F1 - result_direct) / inintial_un


ax2 = fig.add_subplot(212)
ax2.vlines(Nv, 0, np.max(error), color="black", linestyles="dashed")
ax2.vlines(Nv*(Nx-1), 0, np.max(error), color="black", linestyles="dashed")
ax2.vlines(Nv*Nx, 0, np.max(error), color="black", linestyles="dashed")

plt.plot(error, "ro--")
ax2.set_xlabel(r"$n$", fontsize=FontSize)
ax2.set_ylabel(r"$\frac{u_n^{F1} - u_n^{direct}}{u_n^{0}}$", fontsize=FontSize)

plt.tight_layout()
plt.show()

# plt.show()
