import numpy as np
import argparse
import yaml
import pickle
import pandas as pd
from tqdm import tqdm
import imageio

CONFIG = yaml.safe_load(open("configs/global.yml"))


def read_config_matrices(simulation: str) -> tuple:

    dynamic = imageio.v2.imread(f"configs/{simulation}/dynamic.png")
    dynamic = dynamic[:, :, 0] // 255

    mass = imageio.v2.imread(f"configs/{simulation}/mass.png")
    mass = mass[:, :, 0] / 255

    amplitude = imageio.v2.imread(f"configs/{simulation}/amplitude.png")
    amplitude = amplitude[:, :, 0] / 255

    elastic_constant = imageio.v2.imread(
        f"configs/{simulation}/elastic_constant.png")
    elastic_constant = elastic_constant[:, :, 0] / 255

    natural_length = imageio.v2.imread(
        f"configs/{simulation}/natural_length.png")
    natural_length = natural_length[:, :, 0] / 255

    return dynamic, mass, amplitude, elastic_constant, natural_length


def _calculate_forces(pos: np.ndarray,
                      elastic_constant: float,
                      natural_length: float,
                      particle_separation: float,
                      ) -> np.ndarray:
    """
    Calculate a 2D array with the forces of a given particle in the vertical
    direction. Element `[i, j]` of the array is the force acting on particle
    in the position `(i, j)` of the mesh in the z-axis (vertical).

    Parameters
    ----------
    pos : np.ndarray
        The z-positions of the particles.
    elastic_constant : float
        The elastic constant of the springs in N/m.
    natural_length : float
        The natural length of the springs in m.
    particle_separation : float
        The interparticle separation in m.

    Returns
    -------
    forces : np.ndarray
        A 2D array with the forces of the particles.
    """
    ny, nx = pos.shape
    displacements = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    # Calculate the elastic forces
    forces = np.zeros((ny, nx), dtype=np.float64)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            for dx, dy in displacements:
                length = np.sqrt(
                    particle_separation**2
                    + (pos[i, j] - pos[i + dx, j + dy])**2)
                elongation = length - natural_length
                forces[i, j] += - elastic_constant \
                    * elongation \
                    * (pos[i, j] - pos[i + dx, j + dy]) / length

    return forces


def simulate(masses: np.ndarray,
             initial_pos: np.ndarray,
             particle_types: np.ndarray,
             elastic_constant: float,
             natural_length: float,
             particle_separation: float,
             timestep: float,
             n_steps: int,
             filename: str,
             ) -> tuple:
    """
    Simulate the dynamics of mesh.

    Parameters
    ----------
    masses : np.ndarray
        A list with the masses of the particles to be simulated.
    initial_xposs : np.ndarray
        A list with the initial positions of the particles in the x-axis.
    particle_types : list
        A list with `0` for the static particles and `1` for the dynamic ones.
    elastic_constant : float
        The elastic constant of the springs in N/m.
    natural_length : float
        The natural length of the springs in m.
    particle_separation : float
        The interparticle separation in m.
    timestep : float
        The timestep of the simulation.
    n_steps : int
        The number of steps to simulate.
    filename : str
        The name of the simulation file.

    Returns
    -------
    pd.DataFrame
        A Pandas data frame with the results of the simulation.
    """

    time = np.zeros(n_steps)
    pos = np.zeros((n_steps, initial_pos.shape[0], initial_pos.shape[1]))
    vel = np.zeros((n_steps, initial_pos.shape[0], initial_pos.shape[1]))
    pos[0] = initial_pos
    pos[:, particle_types == 0] = 0  # Fix static particles to initial pos

    # Integrate using the leapfrog method
    for step in tqdm(range(1, n_steps),
                     desc="Integrating...",
                     colour="#A241B2",
                     ncols=100):

        # Calculate the forces acting on the particles on previous step
        forces_then = _calculate_forces(
            pos=pos[step - 1],
            elastic_constant=elastic_constant,
            natural_length=natural_length,
            particle_separation=particle_separation)

        # Update the positions
        acc_then = forces_then / masses
        pos[step] = pos[step - 1] + vel[step - 1] * timestep \
            + 0.5 * acc_then * timestep**2
        pos[:, particle_types == 0] = 0

        # Recalculate the force in the current step
        forces_now = _calculate_forces(
            pos=pos[step],
            elastic_constant=elastic_constant,
            natural_length=natural_length,
            particle_separation=particle_separation)

        # Update the velocities
        acc_now = forces_now / masses
        vel[step] = vel[step - 1] + 0.5 * (acc_then + acc_now) * timestep
        vel[:, particle_types == 0] = 0

        # Update the time
        time[step] = time[step - 1] + timestep

    time = pd.DataFrame({"Time": time})

    # Round output values
    time = time.round(decimals=CONFIG["DATAFRAME_DECIMALS"])
    pos = np.round(pos, CONFIG["DATAFRAME_DECIMALS"])

    # Sample data to a given FPS
    n_rows = int(CONFIG["FPS"] * time["Time"].to_numpy()[-1])
    idx = (np.linspace(
        0, 1, n_rows, endpoint=False) * len(time)).astype(np.int64)
    time = time.iloc[idx]
    time.reset_index(inplace=True, drop=True)
    pos = pos[idx]

    time.to_csv(f"results/{filename}/time.csv")
    output = open(f"results/{filename}/data.pkl", "wb")
    pickle.dump(pos, output)
    output.close()

    return time, pos


def main():
    # Get IC file name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation", type=str, required=True,
        help="The name of the folder that contains the configuration files.")
    args = parser.parse_args()

    # Read configuration file
    dynamic, mass, amplitude, _, _ = read_config_matrices(
        simulation=args.simulation)
    PHYSICS = yaml.safe_load(
        open(f"configs/{args.simulation}/physics.yml"))

    # Run simulation
    simulate(masses=mass * PHYSICS["MASS"],
             initial_pos=amplitude * PHYSICS["AMPLITUDE"],
             particle_types=dynamic,
             elastic_constant=PHYSICS["ELASTIC_CONSTANT"],
             natural_length=PHYSICS["NATURAL_LENGTH"],
             particle_separation=PHYSICS["PART_SEPARATION"],
             timestep=PHYSICS["TIMESTEP"],
             n_steps=PHYSICS["N_STEPS"],
             filename=args.simulation)


if __name__ == "__main__":
    main()
