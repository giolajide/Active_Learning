"""
This script runs MD using an MLIP

Sept 2025: Written 8 months ago; haven't used since.
Can it still be useful?
"""
from argparse import ArgumentParser
from sys import argv, exit
#from nequip.ase import NequIPCalculator
import numpy as np
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import read, Trajectory, write
from ase.optimize.fire import FIRE
from ase.units import fs
from os import makedirs, getcwd, path
from typing import List
from ase import Atoms
import warnings
import inspect
#import time

#custom script in the current directory
from md_quality_assurance import (
        check_for_explosion, analyze_neighbors_evolution,
        check_if_equilibrated, check_for_overlapping_atoms,
        check_system_integrity, check_for_floating_atoms,
        SimulationError, MULTIPLIER,
        EXPANSION_THRESHOLD, EQUILIBRATION_TOLERANCE, MODE,
        EQUILIBRATION_WINDOW, OVERLAP_CUTOFF,
        MIN_COORDINATION, BOND_SCALING_FACTOR
        )


PICOSECONDS_TO_FEMTOSECONDS = 1e3
STARTING_TEMPERATURE = 300 #K
TIMESTEP = 1.0 #fs
TEMP_STEP = 10 #K
WRITE_INTERVAL = 50 
FRICTION_FACTOR = 0.02 #fs
FMAX = 0.2 #eV/A
OPT_STEPS = 2000
CHECK_FREQUENCY = 10 #ps



def initialize_md(atoms: Atoms, temperature: float):
    """Initialize MD with an M-B distribution of velocities and set the COM stationary"""
    print("initializing MD")
    MaxwellBoltzmannDistribution(atoms = atoms, temperature_K = temperature, force_temp = False)
    Stationary(atoms)


def run_NVT(
        atoms: Atoms, timestep: float, temperature: float,
        friction: float, logfile: str, trajectory: str,
        write_interval: int, steps: int, threshold: int = None
        ):
    """Setup the Langevin Thermostat and run the MD"""
    print("Running NVT")
    with Trajectory(trajectory, "a", atoms = atoms) as traj:
        dyn = Langevin(
                atoms, timestep = timestep, temperature_K = temperature,
                fixcm = False, friction = friction, logfile = logfile
                )
        dyn.attach(traj.write, interval = write_interval)
        dyn.run(steps)

        if threshold is not None:
            to_analyze = read(trajectory, f"{-threshold}:")
            return to_analyze


def preoptimize(atoms: Atoms, fmax: float, save_dir: str, steps: int):
    """Tries to optimize Atoms object"""
    skip_image = False
    opt = FIRE(atoms, logfile = f"{save_dir}/opt.log", trajectory = f"{save_dir}/opt.traj")
    try:
        opt.run(fmax = fmax, steps = steps)
        atoms.write(f"{save_dir}/optimized.traj")
        print("Optimization succeeded")
    except Exception as e:
        print(f"Optimization failed!\n{e}")
        skip_image = True

    return skip_image


def setup_NVT(
        total_time: float, temperatures: List[float], trajectory_file: str, model_path: str,
        ramp_time: float, temp_step: int = TEMP_STEP, start_temp: float = STARTING_TEMPERATURE,
        timestep: float = TIMESTEP, write_interval: int = WRITE_INTERVAL,
        friction_factor: float = FRICTION_FACTOR, check_frequency: float = CHECK_FREQUENCY,
        ):
    """
    Runs NVT on each structure at each temperature, using a specified model
    NOTE!!!!    The current function runs each structure serially. I should eventually make this parallel
                (but then it'll no longer be able to run on one GPU, since multiprocessing's spawn() can't handle multiple CUDA processes well)

    Parameters:
    total_time (float):         The total length of the simulation, in picoseconds
    model_path (str):           Absolute path to your NequIP model
    temperatures (List[float]): List of the temperatures at which to run the NVT, in the format: x y z (i.e. without commas)
    trajectory_file (str):      NVT will be done for each image in this file
    timestep (float):           Timestep for the NVT, in femtoseconds. Optional. Default = 1.0
    write_interval (int):       How frequently to update the Trajectory object. Optional. Default = 50
    friction_factor (float):    Friction factor for the Langevin thermostat. Optional. Default = 0.02 fs-1
    start_temp (float):         Starting temperature for the ramp up, in K. Optional. Default = 300 K
    ramp_time (float):          Time to run at each ramping temp, in picoseconds.
    temp_step (int):            Ramping temperature step size, in K. Optional. Default = 10 K
    check_frequency (float):    How frequently (in ps) should we carry out Quality Assurance checks? Optional. Default = 10 ps.

    Does:
    Creates a directory for each requested temperature for each index of the specified trajectory, and runs the NVT there

    Example usage:
        from Active_Learning.run_md import setup_NVT
        setup_NVT(
        total_time = 1000, temperatures = 100 200 300 400, ramp_time = 10,
        trajectory_file = nanowires.traj, timestep = 2.5, write_interval = 100
        )
    """
    steps = int(total_time * PICOSECONDS_TO_FEMTOSECONDS / timestep)
    ramp_steps = int(ramp_time * PICOSECONDS_TO_FEMTOSECONDS / timestep)
    timestep = timestep * fs
    friction = friction_factor * fs

    try:
        atoms = read(trajectory_file, ":")
        print("Read trajectory file")
    except Exception as r:
        raise RuntimeError(f"Could not read trajectory file {trajectory_file}:\t{r}")

    temperatures.sort()
    first_temperature = temperatures[0]
    former_temperature = first_temperature #we dont start the cranking for each different temp from start_temp but from where the just-preceeding (hence, lower-temperature) run ended

    for index, image in enumerate(atoms):
        print(f"Starting image {index}")
        if not path.exists(model_path):
            raise FileNotFoundError(f"MLIP file not found at {model_path}")

        """Modified on Feb 7th, 2025"""
        try:
            from nequip.ase import NequIPCalculator
            image.calc = NequIPCalculator.from_deployed_model(model_path, device = "cuda")
        except:
            from mace.calculators import mace_mp
            image.calc = mace_mp(
                    model = "medium", #"https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model",
                    dispersion = True
                    )

        skip_image = False

        for temperature in temperatures:
            #Sort out working directories
            save_dir = getcwd() + "/" + str(index) + "/" + str(temperature)
            print(f"Results will be saved to {save_dir}")
            if path.exists(save_dir):
                warnings.warn_explicit(
                        f"{save_dir} already exists. You may be about to overwrite results!",
                        category = UserWarning,
                        filename = __file__,
                        lineno = inspect.currentframe().f_lineno
                        )
            makedirs(save_dir, exist_ok = True)

            if temperature == first_temperature:
                #First, optimize the structure
                skip_image = preoptimize(atoms = image, fmax = FMAX, steps = OPT_STEPS, save_dir = save_dir) 
                temperature_array = np.arange(min(start_temp, first_temperature), temperature, temp_step)
            else:
                temperature_array = np.arange(former_temperature + temp_step, temperature, temp_step)

            if skip_image: #move on to the next image. this run is abandoned
                break

            print(f"To get to {temperature}, we will be gradually heating at the below temperatures:")
            print(" ".join([str(temp) for temp in temperature_array]))

            final_logfile = f"{save_dir}/{temperature}.log"
            final_trajectory = f"{save_dir}/{temperature}.traj"

            ramp_dir = save_dir + "/" + "ramp"
            makedirs(ramp_dir, exist_ok = True)
            ramp_trajectory = f"{ramp_dir}/To_{temperature}.traj"

            #Now, crank the temperatures upward till the desired temperature
            initialize_md(image, temperature = temperature_array[0])


            for incremental_temp in temperature_array:
                print(f"""Running NVT at {incremental_temp} K, on image {index} of {trajectory_file},
                        with {timestep * fs} fs timestep and {friction * fs} fs-1 friction factor,
                        aiming to eventually reach {temperature} K""")
                ramp_logfile = f"{ramp_dir}/{incremental_temp}.log"
                run_NVT(
                        atoms = image, timestep = timestep, temperature = incremental_temp,
                        friction = friction, logfile = ramp_logfile, steps = ramp_steps,
                        write_interval = write_interval, trajectory = ramp_trajectory
                        )
                print("NVT done")

                #Quit the MD immediately if the MLIP shows instability
                if not check_system_integrity(image):
                    skip_image = True
                    print(f"MD unstable, hence abandoned. Going to image: {index + 1}")
                    break
                else:
                    print("MD MIGHT be alright")

            if skip_image: #move on to the next image. this run is abandoned
                break

            #Finally and most importantly, run the MD at the desired temperature
            print(f"\nFinally at {temperature}")
            if check_frequency >= total_time:
                warnings.warn_explicit(
                f"""Checking frequency: {check_frequency} ps>= total simulation time: {total_time} ps
                Reducing it to {total_time // 4} ps""",
                category = UserWarning,
                filename = __file__,
                lineno = inspect.currentframe().f_lineno
                )
                check_frequency = total_time // 4

            iterations = max(int(total_time / check_frequency), 4)
            interval = 4 #dont always check for equilibration
            print(f"We are going to do {iterations - 1} sanity checks and {iterations // interval} checks for equilibration")
            equilibrated = False
            for i in range(iterations - 1):
                if equilibrated:
                    i = max(i, range(iterations - 3)) #if equilibrated, then possibly stop the MD early
                trajectory_to_analyze = run_NVT(
                        atoms = image, timestep = timestep, temperature = temperature,
                        friction = friction, logfile = final_logfile, steps = int(steps / iterations),
                        write_interval = write_interval, trajectory = final_trajectory, threshold = EXPANSION_THRESHOLD
                        )
                #Quit the MD immediately if the MLIP shows instability. Analyze also with the change in the number of neighbors (if first checks pass)
                if not check_system_integrity(image):
                    print(f"NVT for image {index} of {trajectory_file} at {temperature} K has been prematurely ended")
                    break
                elif analyze_neighbors_evolution(trajectory = trajectory_to_analyze):
                    print(f"NVT for image {index} of {trajectory_file} at {temperature} K has been prematurely ended")
                    break
                else:
                    print("MD is almost certainly alright")

                    #check if the system is equilibrated, so we can possibly terminate the MD early
                    if i % interval == 0:
                        possibly_equilibrated = read(final_trajectory, f"{-EQUILIBRATION_WINDOW}:")
                        if check_if_equilibrated(
                                trajectory = possibly_equilibrated, target_temperature = temperature,
                                tolerance = EQUILIBRATION_TOLERANCE, window = len(possibly_equilibrated)
                                ):
                            print("MD is probably equilibrated. We will happily terminate soon . . . ")
                            equilibrated = True

            former_temperature = temperature

            print(f"NVT for image {index} of {trajectory_file} at {temperature} K done!\n\n")



if __name__ == "__main__":
    parser = ArgumentParser(description = "This script is for running an NVT with a specified model")
    parser.add_argument("--temperatures", "-T", type = float, nargs = "+", required = True, help = "List of MD Temperatures in Kelvin. In the format: x y z. Required")
    parser.add_argument("--timestep", "-t", type = float, default = 1.0, help = "Timestep (fs). Default = 1")
    parser.add_argument("--total_time", "-TT", type = float, required = True, help = "Total simulation time at desired temp (ps). Required")
    parser.add_argument("--write_interval", "-W", type = int, default = 50, help = "Interval for writing structures. Default = 50")
    parser.add_argument("--friction_factor", "-f", type = float, default = 0.02, help = "Friction factor for the thermostat (fs). Default = 0.02 fs")
    parser.add_argument("--model_path", type = str, required = True, help = "Absolute path to your NequIP MLIP")
    parser.add_argument("--trajectory", "-tr", type = str, required = True, help = "Structure(s) on which to run the MD. Required")
    parser.add_argument("--temp_step", "-ts", type = int, default = 10, help = "Step size for ramping up temp to the desired temp. Default = 10 K")
    parser.add_argument("--ramp_time", "-RT", type = float, required = True, help = "Simulation time (in ps) at each temp during the ramp. Required")
    parser.add_argument("--start_temp", "-s", type = int, default = 300, help = "Starting temperature for the ramp, in Kelvin. Default = 300 K")
    parser.add_argument("--check_frequency", "-cf", type = float, default = 1, help = "How frequently (in ps) to check the MD for sanity. Optional. Default = 1 ps")

    args = parser.parse_args()

    setup_NVT(total_time = args.total_time, temperatures = args.temperatures, timestep = args.timestep, temp_step = args.temp_step, ramp_time = args.ramp_time, start_temp = args.start_temp, write_interval = args.write_interval, friction_factor = args.friction_factor, trajectory_file = args.trajectory, model_path = args.model_path, check_frequency = args.check_frequency)


