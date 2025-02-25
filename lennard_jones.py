import numpy as np

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
def initialize_particles(N, density):
    """Initialize particle positions and velocities."""
    L = np.sqrt(N / density)  # Box size
    positions = np.random.rand(N, 2) * L  # Random positions in 2D
    velocities = np.random.randn(N, 2)  # Random velocities
    return positions, velocities, L

# Compute forces
def compute_forces(positions, L, sigma=1.0, epsilon=1.0, rcutoff=2.5):
    """Compute forces on all particles."""
    N = len(positions)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            r_vec = r_vec - L * np.round(r_vec / L)  # Minimum image convention
            r = np.linalg.norm(r_vec)
            if r > 0:
                force_magnitude = lj_force(r, sigma, epsilon, rcutoff)
                forces[i] -= force_magnitude * r_vec / r
                forces[j] += force_magnitude * r_vec / r
    return forces

# Update positions and velocities using Leapfrog algorithm
def update_positions_velocities(positions, velocities, forces, dt, L):
    """
    Update positions and velocities using the Leapfrog algorithm.
    """
    # Update velocities at the half-step
    velocities += 0.5 * forces * dt

    # Update positions at the full step
    positions += velocities * dt

    # Apply periodic boundary conditions
    positions = positions % L

    # Recompute forces at the new positions
    new_forces = compute_forces(positions, L)

    # Update velocities at the full step
    velocities += 0.5 * new_forces * dt

    return positions, velocities, new_forces

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
        for pos in positions:
            f.write(f"X {pos[0]} {pos[1]} 0.0\n")  # Xi x y z

# Main simulation loop performing time integration
def simulate(N=20, density=1.0, dt=0.001, steps=1000):
    """Run the simulation."""
    # Initialize particles
    positions, velocities, L = initialize_particles(N, density)

    filename = "lj_trajectory.xyz"
    
    # Clear the .xyz file before starting
    with open(filename, "w") as f:
        pass
    
    # Compute initial forces
    forces = compute_forces(positions, L)
    
    # Simulation loop
    for step in range(steps):
        positions, velocities, forces = update_positions_velocities(positions, velocities, forces, dt, L)
        save_xyz(positions, step, filename)
    
    print(f"Trajectory saved to {filename}")

# Run the simulation
if __name__ == "__main__":
    #density is the number of particles per unit area. Inverse relationship with box size
    simulate(N=20, density=1.0, dt=0.00000035, steps=5000)