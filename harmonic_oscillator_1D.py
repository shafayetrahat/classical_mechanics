import numpy as np
import matplotlib.pyplot as plt

# _____________________________variables________________________________________________
A               = 0
x1              = 0     # position of particle nr 1, initially A = 0
x2              = 0     # position of particle nr 2, initially A = 0
x_equilibrium   = 1     # equilibrium position
spring_constant = 1     # spring constant
mass            = 1     # particle mass
analysis_time   = 50     # total simulation time
timesteps       = 1000    # nr of timesteps
dt              = analysis_time/timesteps  # timestep length

# random velocities in the interval (-1:1), should velocities be equal to each other? 
v1 = np.random.uniform(-1, 1)  
v2 = -v1
print("Initial velocity of the particles = " + f"{v1:.3f}.")


# ______________________________functions________________________________________________
def calculate_force(x1, x2):
    force = -spring_constant * (x1 - x2 - x_equilibrium)
    return force

def calculate_momentum(mass, v1, v2):
    # calculate linear momentum, , for one particle p=m*v 
    momentum = mass * (v1 + v2)
    return momentum

def calculate_velocity(v1_old, v2_old, dt, force, mass):
    # Compute velocity, v(t+Δt/2) = v(t-Δt/2) + a(t) * Δt,  a = F/m
    v1_new = v1_old + 0.5 * dt * (force / mass)
    v2_new = v2_old - 0.5 * dt * (force / mass)
    return v1_new, v2_new

def calculate_position(x1_old, x2_old, dt, v1_halfstep, v2_halfstep):
    # Compute position, x(t+Δt) = x(t-Δt/2) + v(t+Δt/2) * Δt
    x1_new = x1_old + v1_halfstep * dt
    x2_new = x2_old + v2_halfstep * dt
    return x1_new, x2_new

def save_xyz(filename, x1_list, x2_list):
    """
    Saves the trajectory of two particles in .xyz format.
    
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
            f.write(f"P2 {x2_list[i]:.6f} 0.000000 0.000000\n")  # Particle 
# _________________________________store data_____________________________________________
# create lists for data storing
x1_list, x2_list = [x1], [x2]
v1_list, v2_list = [v1], [v2]
# create linear momentum list
momentum_list = [calculate_momentum(mass, v1, v2)]


# ______________________________leap-frog algorithm_______________________________________
# Leap-frog algorithm: acceleration and location at full timesteps and velocity at half time steps
force = calculate_force(x1, x2)
for i in range(timesteps):
    v1_halfstep, v2_halfstep = calculate_velocity(v1, v2, dt, force, mass)
    x1, x2 = calculate_position(x1, x2, dt, v1_halfstep, v2_halfstep)
    force  = calculate_force(x1, x2)
    v1, v2 = calculate_velocity(v1_halfstep, v2_halfstep, dt, force, mass)

    # Store data
    x1_list.append(x1)
    x2_list.append(x2)
    v1_list.append(v1)
    v2_list.append(v2)
    momentum_list.append(calculate_momentum(mass, v1, v2))

# ___________________________analytical solution__________________________________________
# # Angular frequency
# omega = np.sqrt(spring_constant / mass)
# # The solution for the relative displacement between the particles
# t_values = np.linspace(0, analysis_time, timesteps + 1)
# x_relative = A * np.cos(omega * t_values)
# x1_analytical = 0.5 * x_relative
# x2_analytical = -0.5 * x_relative

# ________________________check linear momentum conservation______________________________
print(f"Initial momentum: {momentum_list[0]:.3f}")
print(f"Final momentum: {momentum_list[-1]:.3f}")
print(f"Momentum change: {momentum_list[-1] - momentum_list[0]:.3e}")  # Should be close to zero


# ____________________________plot results________________________________________________
# Position plot
plt.figure(figsize=(10, 4))
plt.plot(range(timesteps+1), x1_list, label="Particle 1", color='#1f77b4', lw=1)
plt.plot(range(timesteps+1), x2_list, label="Particle 2", color='#ff7f0e', lw=1)
# plt.plot(range(timesteps+1), x1_analytical, label="Particle 1", color='#1f77b4', linestyle="--", lw=2)
# plt.plot(range(timesteps+1), x2_analytical, label="Particle 2", color='#ff7f0e', linestyle="--", lw=2)
plt.xlabel("Timestep")
plt.ylabel("Position")
# Other parameter values that should be visible on the plot?
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