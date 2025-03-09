import numpy as np
import matplotlib.pyplot as plt
import os

# Lennard-Jones potential and force
def lj_potential(r, sigma=1.0, epsilon=1.0, rcutoff=2.5):
    """Lennard-Jones potential with cutoff."""
    if r < rcutoff:
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    else:
        return 0.0

def lj_force(r, sigma=1.0, epsilon=1.0, rcutoff=2.5):
    """Lennard-Jones force with cutoff."""
    if r < rcutoff:
        return 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
    else:
        return 0.0

# Initialize particle positions and velocities
def initialize_particles(N, density=1.0, desired_temperature=298.0):
    """Initialize particle positions and velocities."""
    L = np.sqrt(N / density)  # Box size
    
    # Initialize positions uniformly within the box, centered around the box center
    margin = 0.1 * L  # 10% margin from the edges
    positions = (L / 2 - margin) + 2 * margin * np.random.rand(N, 2)
    
    # Initialize velocities with small random values
    velocities = np.random.randn(N, 2)  # Random velocities
    
    # Scale velocities to match the desired temperature
    kinetic_energy = 0.5 * np.sum(velocities**2)
    desired_kinetic_energy = 1.5 * N * desired_temperature  # 3/2 N k_B T (k_B = 1)
    scaling_factor = np.sqrt(desired_kinetic_energy / kinetic_energy)
    velocities *= scaling_factor
    
    return positions, velocities, L

# Compute forces
def compute_forces(positions, L, sigma=1.0, epsilon=1.0, rcutoff=2.5):
    """Compute forces on all particles."""
    N = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            r_vec = r_vec - L * np.round(r_vec / L)  # Minimum image convention
            r = np.linalg.norm(r_vec)
            if r > 0:
                force_magnitude = lj_force(r, sigma, epsilon, rcutoff)
                forces[i] -= force_magnitude * r_vec / r
                forces[j] += force_magnitude * r_vec / r
                potential_energy += lj_potential(r, sigma, epsilon, rcutoff)
    return forces, potential_energy

# Update positions and velocities using Leapfrog algorithm
def update_positions_velocities(positions, velocities, forces, dt, L, use_pbc=True):
    """
    Update positions and velocities using the Leapfrog algorithm.
    """
    # Update velocities at the half-step
    velocities += 0.5 * forces * dt

    # Update positions at the full step
    positions += velocities * dt

    # Apply periodic boundary conditions if enabled
    if use_pbc:
        positions = positions % L

    # Recompute forces at the new positions
    new_forces, potential_energy = compute_forces(positions, L)

    # Update velocities at the full step
    velocities += 0.5 * new_forces * dt

    return positions, velocities, new_forces, potential_energy

# Save positions to .xyz file
def save_xyz(positions, step, filename="lj_trajectory.xyz"):
    """
    Save particle positions to an .xyz file.
    Each particle's position is saved in the format:
        Xi x y z
    where `i` is the particle index (starting from 1), and z = 0.0 for 2D simulations.
    """
    with open(filename, "a") as f:
        # Write the number of particles
        f.write(f"{len(positions)}\n")
        # Write the step number as a comment line
        f.write(f"Step {step}\n")
        # Write each particle's position with a unique label (X1, X2, X3, ...)
        for i, pos in enumerate(positions, start=1):
            f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")  # Xi x y z

# Plot and save particle positions
def plot_positions(positions, L, step, use_pbc, save_path):
    """Plot particle positions and save the image."""
    plt.figure(figsize=(6, 6))
    plt.scatter(positions[:, 0], positions[:, 1], s=50, c="blue", edgecolors="black")
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.title(f"Step {step} ({'PBC' if use_pbc else 'No PBC'})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Calculate total energy
def calculate_total_energy(positions, velocities, L):
    """Calculate total energy (kinetic + potential)."""
    forces, potential_energy = compute_forces(positions, L)
    kinetic_energy = 0.5 * np.sum(velocities**2)
    return kinetic_energy + potential_energy

# Main simulation loop
def simulate(N=20, density=0.8, dt=0.001, steps=5000, use_pbc=True, desired_temperature=298.0, filename="lj_trajectory.xyz"):
    """Run the simulation."""
    # Initialize particles
    positions, velocities, L = initialize_particles(N, density, desired_temperature)
    
    # Clear the .xyz file before starting
    with open(filename, "w") as f:
        pass
    
    # Compute initial forces
    forces, potential_energy = compute_forces(positions, L)
    
    # Lists to store energy values for plotting
    time_steps = []
    total_energies = []
    save_xyz(positions, 0, filename)
    image_dir = "images"
    if use_pbc != True:
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"step_0_no_pbc.png")
        plot_positions(positions, L, 0, use_pbc, image_path)
        print(f"Images saved to {image_dir}")
    
    # Simulation loop
    for step in range(steps):
        positions, velocities, forces, potential_energy = update_positions_velocities(
            positions, velocities, forces, dt, L, use_pbc
        )
        save_xyz(positions, step+1, filename)
        
        # Calculate and store total energy
        total_energy = calculate_total_energy(positions, velocities, L)
        time_steps.append(step * dt)
        total_energies.append(total_energy)
        
        # Save an image of the particle positions at specific steps
        if use_pbc != True:
            if step % 50 == 0:  # Save every 50 steps
                image_path = os.path.join(image_dir, f"step_{step+1}_no_pbc.png")
                plot_positions(positions, L, step+1, use_pbc, image_path)
    print(f"Trajectory saved to {filename}")
    return total_energies, time_steps

# Run the simulation with and without PBCs
if __name__ == "__main__":
    print("Simulating without PBCs...")
    total_energies_no_pbc, time_steps_no_pbc = simulate(N=20, dt=0.0005, steps=5000, use_pbc=False, filename="lj_trajectory_no_pbc.xyz")
    plt.plot(time_steps_no_pbc, total_energies_no_pbc, label="Total Energy (No PBC)")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy Conservation (No PBC)")
    plt.legend()
    plt.savefig("energy_no_pbc.png")
    plt.clf()

    print("Simulating with PBCs...")

    total_energies_pbc, time_steps_pbc = simulate(N=20, density=0.0006 , dt=0.002, steps=5000, use_pbc=True, filename=f"lj_trajectory_pbc.xyz")
    plt.plot(time_steps_pbc, total_energies_pbc, label="Total Energy (PBC)")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy Conservation (PBC)")
    plt.legend()
    plt.savefig(f"energy_pbc.png")
   
    # plt.show()