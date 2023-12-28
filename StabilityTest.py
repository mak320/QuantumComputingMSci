import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable

"""Paramters"""
xmax = 2 * np.pi   # Spatial domain length
Nx = 100          # Number of spatial grid points
vmax = 10.0       # Maximum velocity magnitude
Nv = 100         # Number of velocity grid points
q_electron = -1.0  # charge of electron
q_ion = 1.0 # charge of ion
epsilon_0 = 1 # Permittivity of free space
tau_c = 0.1      # Collision time
dt = 0.01          # Time step
Nt = 100   # Number of time steps
temperature = 1  # Temperature parameter in bolzmann distribution
m = 1.0         # Mass of the particle

# Spatial grid
dx = 2*xmax / (Nx-1)
x = np.linspace(-xmax/2, xmax/2, Nx, endpoint=True)
print(dx-(x[1]-x[0]))

# Velocity grid
dv = 2 * vmax / (Nv-1)
v = np.linspace(-vmax, vmax, Nv, endpoint=True)


# Maxwellian distribution function in velocity space in one spatial dimension the maxwellian is just the Boltzmann distribution
def Boltzmann(v, temperature, m): 
    normalization = (m / (2 * np.pi * temperature)) ** (3 / 2)
    return normalization * 4 * np.pi * np.exp(-m * v**2 / (2 * temperature))

# Initial distribution function (Gaussian in space and bolzamnn distribution in velocity)
def initial_distribution(x, v, temperature, m):
    mu = 0
    sigma = xmax/8
    
    # initial = np.zeros((len(x), len(v)))
    
    # for i, xval in enumerate(x) :
    #     for j, vval in enumerate(v):
    #         spatial_part = np.exp(-0.5 * ((xval - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    #         velocity_part = Bolzmann(vval, temperature, m)
    #         initial[i, j] = spatial_part * velocity_part
        
    # plt.imshow(initial, origin='lower', extent=[-vmax, vmax, 0, Lx], aspect='auto', cmap='viridis')
    # return initial

    spatial_part = np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    velocity_part = Boltzmann(v, temperature, m)
    
    initial = np.outer(spatial_part, velocity_part)

    # Normalize the distribution function to have total sum equal to 1
    initial /= np.sum(initial) * dx * dv
    
    return initial
    
# Initialize distribution function
f = np.zeros((Nx, Nv, Nt))

# Set initial condition using combined distribution function
f[:, :, 0] = initial_distribution(x, v, temperature, m)

# Electric field (initialized based on initial condition)
E = -(q_electron * np.trapz(f[:, :, 0],  axis=1)  + q_ion / Nx ) 


# Time-stepping loop
for n in range(1, Nt):
    # Update electric field
    E = E - (dt * q_electron / epsilon_0) * dv * np.trapz(v[:, np.newaxis] * f[:, :, n-1], axis=1)
    
    # Finite difference update for distribution function
    f[:, :, n] = f[:, :, n-1] - dt / dx * v[:, np.newaxis] * (np.roll(f[:, :, n-1],-1,axis=0) - np.roll(f[:, :, n-1], 1, axis=0)) / 2 \
                  - (dt * q_electron / m) * E * (np.roll(f[:, :, n-1], -1, axis=1) - np.roll(f[:, :, n-1], 1, axis=1)) / (2 * dv)

    # Set fixed boundary conditions in velocity
    f[:, 0, n] = 0
    f[:, -1, n] = 0    

    # Collision operator
    f[:, :, n] = f[:, :, n] + dt / tau_c * (Boltzmann(v, temperature, m) - f[:, :, n])   



"""Plotting and Animation"""

# Create the figure and axis for the animation
fig, ax = plt.subplots()

# Define the update function for the animation
def init():
    im = ax.imshow(f[:, :, 0], origin='upper', extent=[-vmax, vmax, -xmax, xmax], aspect='auto', cmap='viridis')
    return im,

# Define the update function for the animation
def update(frame):
    ax.clear()
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Spatial Position')
    
    im = ax.imshow(f[:, :, frame], origin='lower', extent=[-vmax, vmax, -xmax, xmax], aspect='auto', cmap='viridis')
    
    # Add text annotation for the timestep
    ax.text(0.02, 0.95, f'Time = {frame}', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color='white')  
    return im,

slowing = 4
# Create the animation
ani = FuncAnimation(fig, update, frames=Nt, interval=dt * Nt * slowing)

# Display the animation
plt.show()



fig = plt.figure(figsize=(8, 8))

cmap = plt.get_cmap('viridis')  # You can choose any colormap you prefer
norm = plt.Normalize(0, Nt-1)  # Normalize based on the number of timesteps

# Subplot for Distribution Function in Real Space
ax1 = fig.add_subplot(211)
for n in range(0, Nt, 1):
    color = cmap(norm(n))
    ax1.plot(x, f[:, 2*len(v)//3, n], label=f'Time step {n}', color=color)
ax1.set_title("distribution function sampled middle of domain")
ax1.set_xlabel('Position')
ax1.set_ylabel('Distribution Function')
ax1.grid()
# ax1.legend()

# Subplot for Distribution Function in Velocity Space
ax2 = fig.add_subplot(212)
for n in range(0, Nt, 1):
    color = cmap(norm(n))
    ax2.plot(v, f[2*len(x)//3, :, n], label=f'Time step {n}', color=color)

ax2.set_xlabel('Velocity')
ax2.set_ylabel('Distribution Function')
ax2.grid()
# ax2.legend()

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # empty array for the colorbar
cbar = fig.colorbar(sm, label='Time Step')

# Display the plot
plt.show() 