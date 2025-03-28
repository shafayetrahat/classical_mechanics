import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time  # for timing the simulation

def parse_arguments():
    """Read in and validate the user input from the command line"""
    parser = argparse.ArgumentParser(
        description="Leap-frog simulation of two partiles harmonically oscillating.",
        usage="python %(prog)s --time <simulation_time> --dt <time_step (dt)>")
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
    return args.time, args.dt

def calculate_force(x1, x2, spring_constant, x_equilibrium):
    """Calculate the force acting between the two particles due to the spring."""
    return -spring_constant * (x1 - x2 - x_equilibrium)

def calculate_momentum(mass, v1, v2):
    """Calculate the linear momentum of the system."""
    return mass * (v1 + v2)

def calculate_velocity(v1_old, v2_old, dt, force, mass):
    """Compute velocity using the leap-frog method."""
    v1_new = v1_old + 0.5 * dt * (force / mass)
    v2_new = v2_old - 0.5 * dt * (force / mass)
    return v1_new, v2_new

def calculate_position(x1_old, x2_old, dt, v1_halfstep, v2_halfstep):
    """Compute new position using the leap-frog method."""
    x1_new = x1_old + v1_halfstep * dt
    x2_new = x2_old + v2_halfstep * dt
    return x1_new, x2_new

def analytical_solution(t, x1_0, x2_0, v1_0, v2_0, mass, spring_constant, x0=0.0):
    """Compute the analytical solution for two identical masses connected by a spring."""
    # Reduced mass for two equal masses
    reduced_m = mass / 2  
    # Angular frequency using the reduced mass
    omega = np.sqrt(spring_constant / reduced_m)

    # Relative motion initial conditions
    x_rel_0 = x1_0 - x2_0 - x0  # Initial relative displacement
    v_rel_0 = v1_0 - v2_0       # Initial relative velocity

    # Amplitude and phase angle
    A = np.sqrt(x_rel_0**2 + (v_rel_0 / omega)**2)
    phi = np.arctan2(-v_rel_0, omega * x_rel_0)

    # Compute relative motion
    x_rel_t = A * np.cos(omega * t + phi)
    # Center of mass motion (since both masses are equal, it simplifies)
    x_cm_t = (x1_0 + x2_0) / 2 + (v1_0 + v2_0) * t / 2
    
    # Compute individual positions
    x1_analytical = x_cm_t + (x_rel_t + x0) / 2
    x2_analytical = x_cm_t - (x_rel_t + x0) / 2
    return x1_analytical, x2_analytical

def calculate_rmsd(x1_sim, x2_sim, x1_analytical, x2_analytical):
    """Calculate the Root Mean Square Deviation (RMSD) between simulated and analytical positions."""
    return np.sqrt(np.mean((x1_sim - x1_analytical)**2 + (x2_sim - x2_analytical)**2))

def leapfrog(timesteps, dt, mass, spring_constant, x_equilibrium, x1, x2, v1, v2):
    """Run the leap-frog simulation."""
    
    # Data storage
    x1_list, x2_list = [x1], [x2]
    v1_list, v2_list = [v1], [v2]
    x1_analytic_list, x2_analytic_list = [x1], [x2]
    momentum_list = [calculate_momentum(mass, v1, v2)]
    rmsd_list = []

    # Leap-frog algorithm
    force = calculate_force(x1, x2, spring_constant, x_equilibrium)
    for step in range(timesteps):
        # Update velocities and positions
        v1_halfstep, v2_halfstep = calculate_velocity(v1, v2, dt, force, mass)
        x1, x2 = calculate_position(x1, x2, dt, v1_halfstep, v2_halfstep)
        force = calculate_force(x1, x2, spring_constant, x_equilibrium)
        v1, v2 = calculate_velocity(v1_halfstep, v2_halfstep, dt, force, mass)

        # Store data
        x1_list.append(x1)
        x2_list.append(x2)
        v1_list.append(v1)
        v2_list.append(v2)
        momentum_list.append(calculate_momentum(mass, v1, v2))

        # Compute analytical solution
        t = (step + 1) * dt
        x1_analytical, x2_analytical = analytical_solution(t, x1_list[0], x2_list[0], v1_list[0], v2_list[0], mass, spring_constant, x_equilibrium)
        x1_analytic_list.append(x1_analytical)
        x2_analytic_list.append(x2_analytical)
        # Calculate RMSD
        rmsd = calculate_rmsd(x1, x2, x1_analytical, x2_analytical)
        rmsd_list.append(rmsd)

    return x1_list, x2_list, momentum_list, rmsd_list, x1_analytic_list, x2_analytic_list

def plot_results(timesteps, x1_list, x2_list, momentum_list, rmsd_list, x1_analytic_list, x2_analytic_list, dt):
    """Plot the results: particle positions, momentum conservation, and RMSD."""
    
    # Position plot
    plt.figure(figsize=(10, 4))
    # leap-frog results
    plt.plot(np.arange(timesteps + 1) * dt, x1_list, label="Leap-frog: Particle 1", color='lightcoral', lw=3)
    plt.plot(np.arange(timesteps + 1) * dt, x2_list, label="Leap-frog: Particle 2", color='aqua', lw=3)
    # analytical results
    plt.plot(np.arange(timesteps + 1) * dt, x1_analytic_list, label="Analytical: Particle 1", color='darkred', lw=1, linestyle='dashed')
    plt.plot(np.arange(timesteps + 1) * dt, x2_analytic_list, label="Analytical: Particle 2", color='midnightblue', lw=1, linestyle='dashed')

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(f"Particle Positions (dt = {dt})")
    plt.legend(loc='upper right')
    plt.xlim(0, timesteps * dt)
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"position_plot_dt_{dt}.png", dpi=300)

    # Momentum plot
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(timesteps + 1) * dt, momentum_list, label="Linear Momentum")
    plt.xlabel("Time")
    plt.ylabel("Linear Momentum")
    plt.title(f"Momentum Conservation (dt = {dt})")
    plt.legend()
    plt.xlim(0, timesteps * dt)
    plt.tight_layout()
    plt.savefig(f"momentum_plot_dt_{dt}.png", dpi=300)

    # RMSD plot
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(timesteps) * dt, rmsd_list, label="RMSD", color='red')
    plt.xlabel("Time")
    plt.ylabel("RMSD")
    plt.title(f"Root Mean Square Deviation (RMSD) (dt = {dt})")
    plt.legend()
    plt.xlim(0, timesteps * dt)
    plt.tight_layout()
    plt.savefig(f"rmsd_plot_dt_{dt}.png", dpi=300)

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
            f.write(f"P2 {x2_list[i]:.6f} 0.000000 0.000000\n")  # Particle 2

if __name__ == "__main__":
    # _____________________________variables________________________________________________
    # Read in the user's input: total simulation time and time step length (Δt)
    analysis_time, dt = parse_arguments()
   
    # Fixed variables, not asked from the user   
    mass = 1                 # Particle mass
    spring_constant = 1      # Spring constant
    x_equilibrium = 1        # Equilibrium position

    # rmsd_values = []        # necessary when plotting mean RMSD vs Δt
    # comp_time_values = []   # necessary when plotting computational time vs Δt
    dt_values = [dt]
    for dt in dt_values:
        timesteps = int(analysis_time / dt)
        x1, x2 = 0, 0                           # Initial positions
        v1, v2 = np.random.uniform(-1, 1, 2)    # Initial velocities

        print(f"Running simulation with dt = {dt}")
        # start_time = time.time()  # Start timing, decomment when performing analysis part
        x1_list, x2_list, momentum_list, rmsd_list, x1_analytic_list, x2_analytic_list = leapfrog(
            timesteps, dt, mass, spring_constant, x_equilibrium, x1, x2, v1, v2
        )
        # end_time = time.time()  # End timing, decomment when performing analysis part

        # Store RMSD and computational time, when doing analysis part
        # rmsd_values.append(np.mean(rmsd_list))
        # comp_time_values.append(end_time - start_time)

        print(f"Initial momentum: {momentum_list[0]:.3f}")
        print(f"Final momentum: {momentum_list[-1]:.3f}")
        print(f"Momentum change: {momentum_list[-1] - momentum_list[0]:.3e}")

        # Plot results
        plot_results(timesteps, x1_list, x2_list, momentum_list, rmsd_list, x1_analytic_list, x2_analytic_list, dt)
        save_xyz(f"trajectory_dt_{dt}.xyz", x1_list, x2_list)

    # ___________analysis of computational time vs. Δt and mean RMSD vs. Δt____________________________

    # Plot RMSD vs. Timestep
    # plt.figure(figsize=(8, 5))
    # plt.plot(dt_values, rmsd_values, marker='o', linestyle='-', color='blue')
    # plt.xlabel("Timestep (Δt)")
    # plt.ylabel("Mean RMSD")
    # plt.title("Mean RMSD vs. Timestep")
    # plt.grid(True)
    # plt.savefig("rmsd_vs_timestep.png")
    # plt.show()

    # Plot Computational Time vs. Timestep
    # plt.figure(figsize=(8, 5))
    # plt.plot(dt_values, comp_time_values, marker='o', linestyle='-', color='red')
    # plt.xlabel("Timestep (Δt)")
    # plt.ylabel("Computational Time (s)")
    # plt.title("Computational Time vs. Timestep")
    # plt.grid(True)
    # plt.savefig("time_vs_timestep.png")
    # plt.show()
