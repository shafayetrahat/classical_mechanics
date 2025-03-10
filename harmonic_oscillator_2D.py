import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from mpmath import mp
# Set precision to 50 decimal places
mp.dps = 50

def parse_arguments():
    """Read in and validate the user input from the command line"""
    parser = argparse.ArgumentParser(
        description="Leap-frog simulation of two partiles harmonically oscillating in 2D.",
        usage="python %(prog)s <simulation_time> <time_step (dt)>")
    parser.add_argument("--time", type=float, required=True, help="Total simulation time (must be > 0)")
    parser.add_argument("--dt", type=float, required=True, help="Time step length (must be > 0")
    args = parser.parse_args()

    # Validation of the input
    if args.time <= 0:
        print("Error: Simulation time must be greater than 0.")
        sys.exit(1)
    elif args.dt <= 0:
        print("Error: Time step (dt) must be greater than 0.")
        sys.exit(1)
    elif args.dt > args.time:
        print("Error: Time step is larger than the simulation time.")
        sys.exit(1)
    return mp.mpf(args.time), args.dt

def calculate_force(x1, x2, y1, y2, spring_constant, C):
    """Calculate the force acting between the two particles due to the spring."""
     # calculate distance between particles
    r12 = mp.sqrt((x1 - x2)**2 + (y1 - y2)**2)  
    if r12 > 1e-12:
        # calculate total force at current displacement from equilibrium
        force_mag = -spring_constant * (r12 - C)
        # calculate force in both dimension, unit vector r12 in component form is [(x1-x2)/r12, (y1-y2)/r12]
        force_x = force_mag * (x1 - x2) / r12
        force_y = force_mag * (y1 - y2) / r12     
    else:
        # to avoid division by zero
        force_x, force_y = mp.mpf(0), mp.mpf(0)
    return force_x, force_y

def calculate_lin_momentum(mass, v1_x, v2_x, v1_y, v2_y):
    """Compute total linear momentum (px, py)."""
    return mass * (v1_x + v2_x), mass * (v1_y + v2_y)

def calculate_ang_momentum(mass, x1, x2, y1, y2, v1_x, v2_x, v1_y, v2_y):
    """Compute total angular momentum."""
    return mass * (x1 * v1_y - y1 * v1_x + x2 * v2_y - y2 * v2_x)

def calculate_kinetic_energy(mass, v1_x, v2_x, v1_y, v2_y):
    """Compute the total kinetic energy of the system."""
    # Compute the kinetic energy of each particle
    ke1 = 0.5 * mass * (v1_x**2 + v1_y**2)
    ke2 = 0.5 * mass * (v2_x**2 + v2_y**2)
    # Total kinetic energy
    return ke1 + ke2

def calculate_potential_energy(spring_constant, x1, x2, y1, y2, C):
    """Calculate the potential energy of the spring system."""
    r12 = mp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return 0.5 * spring_constant * (r12 - C) ** 2

def calculate_velocity(v1_x, v2_x, v1_y, v2_y, dt, force_x, force_y, mass):
    """Compute velocity, v(t+Δt/2) = v(t-Δt/2) + a(t) * Δt,  a = F/m"""
    v1_x += 0.5 * dt * (force_x / mass)
    v2_x -= 0.5 * dt * (force_x / mass)
    v1_y += 0.5 * dt * (force_y / mass)
    v2_y -= 0.5 * dt * (force_y / mass)
    return v1_x, v2_x, v1_y, v2_y

def calculate_position(x1, x2, y1, y2, dt, v1_x, v2_x, v1_y, v2_y):
    """Compute new position, x(t+Δt) = x(t) + v(t+Δt/2) * Δt"""
    return x1 + dt * v1_x, x2 + dt * v2_x, y1 + dt * v1_y, y2 + dt * v2_y

def leapfrog(timesteps, dt, mass, spring_constant, C, x1, x2, y1, y2, v1_x, v2_x, v1_y, v2_y):
    """Run the leap-frog simulation."""

    # Data storage
    x1_list, x2_list, y1_list, y2_list = [], [], [], []
    lin_momentum_list, ang_momentum_list, total_energy_list = [], [], []
    kin_energy_list, pot_energy_list = [], []

    # Leap-frog algorithm
    force_x, force_y = calculate_force(x1, x2, y1, y2, spring_constant, C)
    for step in range(int(timesteps)):
        # Leap-frog algorithm: force and location at full timesteps and velocity at half time steps
        v1_x, v2_x, v1_y, v2_y = calculate_velocity(v1_x, v2_x, v1_y, v2_y, dt, force_x, force_y, mass)
        v1_x_halftsep, v2_x_halftsep, v1_y_halftsep, v2_y_halftsep = v1_x, v2_x, v1_y, v2_y  # store half-step data for momentum calc
        x1, x2, y1, y2  = calculate_position(x1, x2, y1, y2, dt, v1_x, v2_x, v1_y, v2_y)
        force_x, force_y = calculate_force(x1, x2, y1, y2, spring_constant, C)        
        v1_x, v2_x, v1_y, v2_y = calculate_velocity(v1_x, v2_x, v1_y, v2_y, dt, force_x, force_y, mass)

        # Store data
        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(y1)
        y2_list.append(y2)
        lin_momentum_list.append(calculate_lin_momentum(mass, v1_x_halftsep, v2_x_halftsep, v1_y_halftsep, v2_y_halftsep))
        ang_momentum_list.append(calculate_ang_momentum(mass, x1, x2, y1, y2, v1_x_halftsep, v2_x_halftsep, v1_y_halftsep, v2_y_halftsep))
        kin_energy_list.append(calculate_kinetic_energy(mass, v1_x, v2_x, v1_y, v2_y))
        pot_energy_list.append(calculate_potential_energy(spring_constant, x1, x2, y1, y2, C))
        total_energy_list.append(kin_energy_list[-1] + pot_energy_list[-1])

    return x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list

def plot_results(timesteps, x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list):
    """Plot the results: particle trajectories, linear and angular momentum conservation."""
    timesteps = int(timesteps)
    # Position plot
    plt.figure(figsize=(8, 8))
    # leap-frog results
    plt.plot(x1_list, y1_list, label="Particle 1", color='blue')
    plt.plot(x2_list, y2_list, label="Particle 2", color='red')
    plt.scatter(x1_list[0], y1_list[0], color='blue', marker='o', label="Start 1")
    plt.scatter(x2_list[0], y2_list[0], color='red', marker='o', label="Start 2")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Particle trajectories")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("2D_trajectory.png")

    # Linear momentum conservation plot
    plt.figure(figsize=(8, 4))
    lin_momentum_array = np.array(lin_momentum_list)
    time = np.linspace(0, float(dt) * timesteps, timesteps)
    plt.plot(time, lin_momentum_array[:, 0], label="Momentum X", color="red")
    plt.plot(time, lin_momentum_array[:, 1], label="Momentum Y", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Momentum")
    plt.title("Linear momentum conservation")
    plt.legend()
    plt.grid()
    plt.xlim(0, max(time))
    plt.savefig("lin_momentum_2D.png")

    # Angular momentum plot
    ang_momentum_array = np.array(ang_momentum_list)
    time = np.linspace(0, float(dt) * timesteps, timesteps)
    plt.figure(figsize=(8, 6))
    plt.plot(time, ang_momentum_array, label="Angular momentum", color="red")
    plt.xlabel("Time")
    plt.ylabel("Angular Momentum")
    plt.title("Angular Momentum Conservation")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max(time))
    plt.tight_layout()
    plt.savefig("ang_momentum_2D.png")

    # Total energy conservation plot
    plt.figure(figsize=(8, 4))
    total_energy_array = np.array(total_energy_list)
    time = np.linspace(0, float(dt) * timesteps, timesteps)
    plt.plot(time, total_energy_array, label="Total energy", color="red")
    plt.xlabel("Time")
    plt.ylabel("Total energy")
    plt.title("Total energy conservation")
    plt.legend()
    plt.grid()
    plt.xlim(0, max(time))
    plt.savefig("total_energy_2D.png")

    # Kinetic and potential energy
    plt.figure(figsize=(8, 4))
    kin_energy_array = np.array(kin_energy_list)
    pot_energy_array = np.array(pot_energy_list)
    plt.plot(range(timesteps), kin_energy_array, label="Kinetic energy", color="red")
    plt.plot(range(timesteps), pot_energy_array, label="Potential energy", color="blue")
    plt.xlabel("Timestep")
    plt.ylabel("Energy")
    plt.title("Kinetic and potential energy")
    plt.legend()
    plt.grid()
    plt.xlim(0, timesteps)
    plt.savefig("kin&pot_energy_2D.png")

def save_xyz(filename, x1_list, x2_list, y1_list, y2_list):
    """
    Saves the trajectory of two particles in .xyz format.##
    
    Parameters:
    filename (str): Name of the output file.
    x1_list (list): List of x-coordinates for particle 1.
    x2_list (list): List of x-coordinates for particle 2.
    y1_list (list): List of y-coordinates for particle 1.
    y2_list (list): List of y-coordinates for particle 2.
    """
    with open(filename, "w") as f:
        for i in range(len(x1_list)):
            f.write("2\n")  # Number of particles
            f.write(f"Timestep {i}\n")  # Comment line
            f.write(f"P1 {float(x1_list[i]):.6f} {float(y1_list[i]):.6f} 0.000000\n")  # Particle 1
            f.write(f"P2 {float(x2_list[i]):.6f} {float(y2_list[i]):.6f} 0.000000\n")  # Particle 2

if __name__ == "__main__":
    # _____________________________variables________________________________________________
    # Read in the user's input: total simulation time and time step length (Δt)
    analysis_time, dt = parse_arguments()
   
    # Fixed variables, not asked from the user   
    mass = mp.mpf(1)                 # Particle mass
    spring_constant = mp.mpf(1)      # Spring constant
    C = mp.mpf(0.5)                  # Equilibrium position
    x1, x2, y1, y2 = mp.mpf(0), mp.mpf(1), mp.mpf(0), mp.mpf(0)  # Initial positions

    # Derived parameters
    timesteps = analysis_time / dt
    random_velocities = np.random.uniform(-1, 1, 4) # Generate random initial velocities
    v1_x, v1_y, v2_x, v2_y = [mp.mpf(v) for v in random_velocities]

    print(f"Initial velocities of particles: v1_x = {str(v1_x)[:6]}, "
        f"v1_y = {str(v1_y)[:6]}, "
        f"v2_x = {str(v2_x)[:6]}, "
        f"v2_y = {str(v2_y)[:6]}")
    
    # Run simulation
    x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, \
    total_energy_list, kin_energy_list, pot_energy_list = leapfrog(
        timesteps, dt, mass, spring_constant, C, 
        x1, x2, y1, y2, 
        v1_x, v2_x, v1_y, v2_y
    )
    
    # For linear momentum
    print(f"Initial linear momentum: px_init = {mp.nstr(lin_momentum_list[0][0], n=3)}, py_init = {mp.nstr(lin_momentum_list[0][1], n=3)}")
    print(f"Final linear momentum: px_fin = {mp.nstr(lin_momentum_list[-1][0], n=3)}, py_fin = {mp.nstr(lin_momentum_list[-1][1], n=3)}")
    print(f"Linear momentum change: d(px) = {mp.nstr(lin_momentum_list[-1][0] - lin_momentum_list[0][0], n=3)}, "
        f"d(py) = {mp.nstr(lin_momentum_list[-1][1] - lin_momentum_list[0][1], n=3)}")

    # For angular momentum
    print(f"Initial angular momentum: {mp.nstr(ang_momentum_list[0], n=3)}")
    print(f"Final angular momentum: {mp.nstr(ang_momentum_list[-1], n=3)}")
    print(f"Angular momentum change: {mp.nstr(ang_momentum_list[-1] - ang_momentum_list[0], n=3)}")

    # Plot results
    plot_results(timesteps, x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list)

    save_xyz("trajectory.xyz", x1_list, x2_list, y1_list, y2_list)