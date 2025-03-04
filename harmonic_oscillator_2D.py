import numpy as np
import matplotlib.pyplot as plt

def calculate_force(x1, x2, y1, y2, spring_constant, C):
    """Calculate the force acting between the two particles due to the spring."""
     # calculate distance between particles
    r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  
    if r12 > 1e-12:
        # calculate total force
        force_mag = -spring_constant * (r12 - C)
        # calculate force in both dimension, unit vector r12 in component form is [(x1-x2)/r12, (y1-y2)/r12]
        force_x = force_mag * (x1 - x2) / r12
        force_y = force_mag * (y1 - y2) / r12     
    else:
        # to avoid division by zero
        force_x, force_y = 0, 0
    return force_x, force_y

def calculate_lin_momentum(mass, v1_x, v2_x, v1_y, v2_y):
    """Compute total linear momentum (px, py)."""
    return mass * (v1_x + v2_x), mass * (v1_y + v2_y)

def calculate_ang_momentum(mass, x1, x2, y1, y2, v1_x, v2_x, v1_y, v2_y):
    """Compute total angular momentum."""
    return mass * (x1 * v1_y - y1 * v1_x + x2 * v2_y - y2 * v2_x)

def calculate_kinetic_energy(mass, v1_x, v2_x, v1_y, v2_y):
    """Compute kinetic energy using center-of-mass and relative motion."""
    # Compute center-of-mass velocity
    v_cm_x = (v1_x + v2_x) / 2
    v_cm_y = (v1_y + v2_y) / 2

    # Compute relative velocity
    v_rel_x = v2_x - v1_x
    v_rel_y = v2_y - v1_y

    # Compute kinetic energy
    ke_cm = mass * (v_cm_x**2 + v_cm_y**2)  # Center-of-mass kinetic energy
    ke_rel = 0.5 * (mass / 2) * (v_rel_x**2 + v_rel_y**2)  # Relative kinetic energy

    return ke_rel + ke_cm

def calculate_potential_energy(spring_constant, x1, x2, y1, y2, C):
    """Calculate the potential energy of the spring system."""
    r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
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
    for step in range(timesteps):
        # Leap-frog algorithm: force and location at full timesteps and velocity at half time steps
        v1_x, v2_x, v1_y, v2_y = calculate_velocity(v1_x, v2_x, v1_y, v2_y, dt, force_x, force_y, mass)
        x1, x2, y1, y2  = calculate_position(x1, x2, y1, y2, dt, v1_x, v2_x, v1_y, v2_y)
        force_x, force_y = calculate_force(x1, x2, y1, y2, spring_constant, C)        
        v1_x, v2_x, v1_y, v2_y = calculate_velocity(v1_x, v2_x, v1_y, v2_y, dt, force_x, force_y, mass)

        # Store data
        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(y1)
        y2_list.append(y2)
        lin_momentum_list.append(calculate_lin_momentum(mass, v1_x, v2_x, v1_y, v2_y))
        ang_momentum_list.append(calculate_ang_momentum(mass, x1, x2, y1, y2, v1_x, v2_x, v1_y, v2_y))
        kin_energy_list.append(calculate_kinetic_energy(mass, v1_x, v2_x, v1_y, v2_y))
        pot_energy_list.append(calculate_potential_energy(spring_constant, x1, x2, y1, y2, C))
        total_energy_list.append(calculate_kinetic_energy(mass, v1_x, v2_x, v1_y, v2_y) + calculate_potential_energy(spring_constant, x1, x2, y1, y2, C))

    return x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list

def plot_results(timesteps, x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list):
    """Plot the results: particle trajectories, linear and angular momentum conservation."""
    
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
    plt.plot(range(timesteps), lin_momentum_array[:, 0], label="Momentum X", color="red")
    plt.plot(range(timesteps), lin_momentum_array[:, 1], label="Momentum Y", color="blue")
    plt.xlabel("Timestep")
    plt.ylabel("Momentum")
    plt.title("Linear momentum conservation")
    plt.legend()
    plt.grid()
    plt.xlim(0, timesteps)
    plt.savefig("lin_momentum_2D.png")

    # Angular momentum plot
    ang_momentum_array = np.array(ang_momentum_list)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    ylims = [(-1, 1), (-1e-13, 1e-13)]  # Different Y-axis limits for each plot to zoom in and out
    for ax, ylim in zip(axes, ylims):
        ax.plot(range(timesteps), ang_momentum_array, label="Angular momentum", color="red")
        ax.grid()
        ax.set_xlim(0, timesteps)
        ax.set_ylim(ylim)
    fig.suptitle("Angular Momentum Conservation")
    fig.supylabel("Angular Momentum")
    fig.supxlabel("Timestep")
    plt.savefig("ang_momentum_2D.png")

    # Total energy conservation plot
    plt.figure(figsize=(8, 4))
    total_energy_array = np.array(total_energy_list)
    plt.plot(range(timesteps), total_energy_array, label="Total energy", color="red")
    plt.xlabel("Timestep")
    plt.ylabel("Total energy")
    plt.title("Total energy conservation")
    plt.legend()
    plt.grid()
    plt.xlim(0, timesteps)
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
            f.write(f"P1 {x1_list[i]:.6f} {y1_list[i]:.6f} 0.000000\n")  # Particle 1
            f.write(f"P2 {x2_list[i]:.6f} {y2_list[i]:.6f} 0.000000\n")  # Particle 2

if __name__ == "__main__":
    # _____________________________variables________________________________________________
    analysis_time = 100       # Total simulation time
    timesteps = 1000         # Number of timesteps
    mass = 1                 # Particle mass
    spring_constant = 1      # Spring constant
    C = 1                    # Equilibrium position

    # Derived parameters
    dt = analysis_time / timesteps
    x1, x2, y1, y2 = 0, 0, 0, 0  # Initial positions
    v1_x, v2_x, v1_y, v2_y = np.random.uniform(-1, 1, 4)  
    print(f"Initial velocities of particles: v1_x = {v1_x:.3f}, " 
          f"v1_y = {v1_y:.3f}, "
          f"v2_x = {v2_x:.3f}, "
          f"v2_y = {v2_y:.3f}")
    
    # Run simulation
    x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list = leapfrog(
        timesteps, dt, mass, spring_constant, C, 
        x1, x2, y1, y2, 
        v1_x, v2_x, v1_y, v2_y)
    
    print(f"Initial linear momentum: px_init = {lin_momentum_list[0][0]:.3f}, py_init = {lin_momentum_list[0][1]:.3f}")
    print(f"Final linear momentum: px_fin = {lin_momentum_list[-1][0]:.3f}, py_fin = {lin_momentum_list[-1][1]:.3f}")
    print(f"Linear momentum change: d(px) = {lin_momentum_list[-1][0] - lin_momentum_list[0][0]:.3e}, "
                                  f"d(py) = {lin_momentum_list[-1][1] - lin_momentum_list[0][1]:.3e}  ")

    print(f"Initial angular momentum: {ang_momentum_list[0]:.3f}")
    print(f"Final angular momentum: {ang_momentum_list[-1]:.3f}")
    print(f"Angular momentum change: {ang_momentum_list[-1] - ang_momentum_list[0]:.3e}")

    # Plot results
    plot_results(timesteps, x1_list, x2_list, y1_list, y2_list, lin_momentum_list, ang_momentum_list, total_energy_list, kin_energy_list, pot_energy_list)

    save_xyz("trajectory.xyz", x1_list, x2_list, y1_list, y2_list)