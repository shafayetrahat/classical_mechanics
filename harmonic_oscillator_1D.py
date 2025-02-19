import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# _____________________________variables________________________________________________   
x_equilibrium   = 1     # equilibrium position
x1              = 0     # position of particle nr 1, initially A = 0
x2              = 0     # position of particle nr 2, initially A = 0
spring_constant = 1     # spring constant
mass            = 1     # particle mass
analysis_time   = 10    # total simulation time
timesteps       = 100   # nr of timesteps
dt              = analysis_time/timesteps  # timestep length
time_array      = np.linspace(0, analysis_time, timesteps+1)

# random velocities in the interval (-1:1)
v1, v2 = np.random.uniform(-1, 1, 2)  

# Store initial state for analytical solution
x1_0 = x1
x2_0 = x2
v1_0 = v1
v2_0 = v2

# ______________________________functions________________________________________________
def calculate_force(x1, x2):
    return -spring_constant * (x1 - x2 - x_equilibrium)

def calculate_momentum(mass, v1, v2):
    # calculate linear momentum, for one particle p=m*v 
    return mass*v1 + mass*v2

def calculate_dv(dt, force, mass):
    # Compute velocity, v(t+Δt/2) = v(t-Δt/2) + a(t) * Δt,  a = F/m
    return dt * (force / mass)

def calculate_dx(dt, v_halfstep):
    # Compute position, x(t+Δt) = x(t) + v(t+Δt/2) * Δt
    return dt * v_halfstep

def save_xyz(filename, x1_list, x2_list):
    """
    Saves the trajectory of two particles in .xyz format.##
    
    Parameters:
    filename (str): Name of the output file.
    x1_list (list): List of x-coordinates for particle 1.
    x2_list (list): List of x-coordinates for particle 2.
    """
    with open(filename, "w") as f:
        for i in range(len(x1_list)):
            f.write("2\n")  # Number of particles
            f.write(f"Timestep {i}\n")  # Comment line
            f.write(f"P1 {x1_list[i]:.6f} 0.000000 0.000000\n")  # Particle 1
            f.write(f"P2 {x2_list[i]:.6f} 0.000000 0.000000\n")  # Particle 2

# input for DE solver, analytical solution
def harmonic_system(t, y):
    x1, v1, x2, v2 = y
    # Force
    F = -spring_constant * (x1 - x2 - x_equilibrium)
    # Equations of motion
    dx1_dt = v1
    dv1_dt = F / mass
    dx2_dt = v2
    dv2_dt = -1 * (F / mass)
    return [dx1_dt, dv1_dt, dx2_dt, dv2_dt]


# _________________________________store data_____________________________________________
# create lists for data storing
x1_list, x2_list = [x1], [x2]
momentum_list = [calculate_momentum(mass, v1, v2)]


# ______________________________leap-frog algorithm_______________________________________
# Leap-frog algorithm: acceleration and location at full timesteps and velocity at half time steps
force = calculate_force(x1, x2)
# Initial half-step velocity calculation
dv = calculate_dv(dt, force, mass)
v1_halfstep = v1 + dv
v2_halfstep = v2 - dv

for i in range(timesteps):
    x1 = x1 + calculate_dx(dt, v1_halfstep)
    x2 = x2 + calculate_dx(dt, v2_halfstep)

    force  = calculate_force(x1, x2)

    dv = calculate_dv(dt, force, mass)
    v1_halfstep = v1_halfstep + dv
    v2_halfstep = v2_halfstep - dv

    # Store data
    x1_list.append(x1)
    x2_list.append(x2)
    momentum_list.append(calculate_momentum(mass, v1_halfstep, v2_halfstep))


# ___________________________analytical solution__________________________________________
# Initial state vector: [x1, v1, x2, v2]
y0 = [x1_0, v1_0, x2_0, v2_0]

# Solve the system of differential equations
solution = solve_ivp(harmonic_system, (0, analysis_time), y0, t_eval=time_array, method='RK45', rtol=1e-8, atol=1e-10)

# Extract the solutions for x1 and x2
x1_sol = solution.y[0]
x2_sol = solution.y[2]


# ____________________________plot results________________________________________________
# Position plot
plt.figure(figsize=(10, 4))
# Leapfrog results
plt.plot(time_array, x1_list, label="Leapfrog particle 1", color='lightsteelblue', lw=1)
plt.plot(time_array, x2_list, label="Leapfrog particle 2", color='pink', lw=1)
# Analytical solution
plt.plot(time_array, x1_sol, label='Analytical particle 1', color='midnightblue', linestyle='--', lw=1)
plt.plot(time_array, x2_sol, label='Analytical particle 2', color='crimson', linestyle='--', lw=1)
plt.xlabel("Timestep")
plt.ylabel("Position")
plt.title("Particle positions")
plt.legend(loc='upper right')
plt.xlim(0, timesteps)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("position_plot.png")

# Momentum plot
plt.figure(figsize=(10, 4))
plt.plot(range(timesteps+1), momentum_list, label="Linear momentum")
plt.xlabel("Timestep")
plt.ylabel("Linear momentum")
plt.title("Momentum conservation")
plt.legend()
plt.xlim(0, timesteps)
plt.tight_layout()
plt.savefig("momentum_plot.png")

save_xyz("trajectory.xyz", x1_list, x2_list)