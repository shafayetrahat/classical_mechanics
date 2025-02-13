import numpy as np
import matplotlib.pyplot as plt

# _____________________________variables________________________________________________
# Intitially, A = 0, same starting position for both particles
x1     = 0     # position of particle nr 1
x2     = 0     # position of particle nr 2
x_equilibrium  = 1     # equilibrium position
timestep       = 1     # time step
spring_constant = 1     # spring constant
mass            = 1     # particle mass
steps           = 10    


# ______________________________functions________________________________________________
def calculate_force(x1, x2):
    force = -spring_constant * (x1 - x2 - x_equilibrium)
    return force

def calculate_momentum(mass, v1, v2):
    momentum = mass * (v1 + v2)
    return momentum

def calculate_velocity(v1_old, v2_old, timestep, force, mass)
    # Compute velocity, v(t+Δt/2) = v(t-Δt/2) + a(t) * Δt,      a = F/m
    v1_new = v1_old + 0.5 * timestep * (force / mass)
    v2_new = v2_old - 0.5 * timestep * (force / mass)
    return v1_new, v2_new

# random velocities in the interval (-1:1)
v1, v2 = np.random.uniform(-1, 1, 2)  

# lists for data storing
x1_list, x2_list = [x1], [x2]
v1_list, v2_list = [v1], [v2]
# momentum for one particle p=m*v 
momentum_list = [calculate_momentum(v1 + v2)]

# Leap-frog algorithm: acceleration and location at full timesteps and velocity at half time steps
for i in range(steps):
    force = calculate_force(x1, x2)
    v1_halfstep, v2_halfstep = calculate_velocity(v1, v2, timestep, force, mass)

    
# Compute position, x(t+Δt) = x(t-Δt/2) + v(t+Δt/2) * Δt
