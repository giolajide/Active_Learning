"""
This script will provide the basis for rationally selecting structures on which to do DFT singlepoints to add to train/valid set

Sept 2025: Not sure how good or bad this script is; never went back to improve it
"""
from nequip.ase import NequIPCalculator
from argparse import ArgumentParser
import torch
import numpy as np
from ase.io import read, Trajectory, write
from os import makedirs, getcwd, path
from typing import List, Tuple
from ase import Atoms
import warnings
import inspect
from tqdm import tqdm
from random import shuffle
import operator
from time import time
from glob import glob

#custom scripts created in this directory
from md_quality_assurance import check_system_integrity
from run_md import (
        WRITE_INTERVAL, TIMESTEP,
        PICOSECONDS_TO_FEMTOSECONDS
        )

SELECTION_TIME_INTERVAL = 5.0               #consider images SELECTION_TIME_INTERVAL picoseconds away from each other
DIVISIONS = 10                              #how many subdivisions of the trajectory (based on system size)
UNCERTAINTY_SELECTION_THRESHOLD = 0.004     #Do not consider any structure for DFT unless its weighted_uncertainty > UNCERTAINTY_SELECTION_THRESHOLD
                                            #TODO: Make this more rigorous, more flexible, less arbitrary
MAX_STRUCTURES_TO_SELECT = 60               #this, I admit, is also arbitrary
                                            #TODO: see above
PERCENTILE = 65                             #consider structures in the 80th percentile (for uncertainties) in their size groups
                                            #TODO: sigh



#warnings.filterwarnings("ignore")

def define_selection_interval(
        md_write_interval: int,
        time_interval: float = SELECTION_TIME_INTERVAL, timestep: float = TIMESTEP
        ) -> int:
    """
    Will return the interval for considering images from the MD for possible selection (for then running DFT singlepoints)

    Parameters:
    md_write_interval (int):    How frequently images were written to the MD trajectory.
                                Optional. Check run_md.py for the default
    time_interval (float):      How long (in ps) between selected images. Optional. Default = 5 (ps)
    timestep (float):           Timestep at which you ran the MD, in fs. Optional. Check run_md.py for the default

    Does:
    Returns the selection interval for taking images from the MD to then consider for possible DFT singlepoints

    Example usage:
        from points_selecter import define_selection_interval
        interval = define_selection_interval(timestep = 2.5)
        atoms = read('md.traj', f'::{interval}')
    """
    if md_write_interval < 1:
        raise ZeroDivisionError(f"Write interval couldn't have been zero. Are you out of your mind?!")

    if time_interval < 1.0:
        warnings.warn_explicit(
        f"""{time_interval} ps as a potential selection time interval might be too short.
        I hope you know what you are doing!""",
        filename = __file__,
        category = UserWarning,
        lineno = inspect.currentframe().f_lineno
        )      

    return int(np.ceil((time_interval * PICOSECONDS_TO_FEMTOSECONDS) // (timestep * md_write_interval)))
        


def committee_singlepoint(atoms: Atoms, model_paths: List[str], calculate_stress: bool = False) -> List[Atoms]:
    """
    Does:
    Runs a singlepoint on the atoms object with each MLIP in the provided committee

    Parameters:
    atoms (Atoms):              Atoms object to be evaluated
    model_paths (List[str]):    Absolute path to each NequIP/Allegro Model
    calculate_stress (bool):     Whether or not to calculate stress. Required. Default = False
    """
    if not torch.cuda.is_available():
        raise RuntimeError(f"{committee_singlepoint.__name__}:\tNo GPU available! Might as well give up right now!")

    evaluated = []
    for model_path in model_paths:
        if not path.exists(model_path):
            raise FileNotFoundError(f"{committee_singlepoint.__name__}:\t{model_path} doesn't exist!")

        new_atoms = atoms.copy()
        calc = NequIPCalculator.from_deployed_model(model_path, device = "cuda")
        new_atoms.calc = calc
        energy = new_atoms.get_potential_energy()

        if calculate_stress:
            stress = new_atoms.get_stresses()
        evaluated.append(new_atoms)
    return evaluated



def calculate_uncertainty(
        trajectory: List[Atoms], energy_coeff: float = 2.0, force_coeff: float = 1.0,
        stress_coeff: float = 0, force_definition: str = "mean"
        ) -> Tuple[float, int]:
    """
    Does:   Calculates standard deviations (a.k.a committee uncertainties) in energies, forces, and stresses (if found)
            This is calculated thus:
                    weighted_uncertainty = eE + fF + sS

    Parameters:
    trajectory (List[Atoms]):    List of atoms object, each having been evaluated by a separate MLIP
    energy_coeff (float):   e, the coefficient for the per_atom_energy uncertainty E (eV/atom). Optional. Default = 2.0
    force_coeff (float):    f, the coefficient for the force loss uncertainty (eV/A). Optional. Default = 1.0
    stress_coeff (float):   s, the coefficient for the stress loss uncertainty (eV/A^2). Optional. Default = 0
    force_definition (str): Should the force uncertainty be based on MAX or MEAN force? Optional. Default = 'max'. Other option: 'mean'
    """
    force_definition = force_definition.lower()
    try:
        if force_definition not in ("max", "mean"):
            raise ValueError(f"Force definition must be 'max' or 'mean'. You said: {force_definition}")
    except ValueError as v:
        print(v)
        print("Defaulting to 'mean'")
        force_definition = "mean"

    uncertainty = []

    try:
        energies = [energy_coeff * atoms.get_potential_energy() / len(atoms) for atoms in trajectory]
        uncertainty.append(np.std(energies))
    except Exception as e:
        raise RuntimeError(f"{calculate_uncertainty.__name__}\tEnergies not found: {e}\nQuitting.") #if no energies, then of course, no stresses nor forces, so, just quit

    if force_definition == "mean":
        fmeans = [force_coeff * (atoms.get_forces() ** 2).sum(axis=1).mean()**(1/2) for atoms in trajectory]
        if fmeans:
            uncertainty.append(np.std(fmeans))
    elif force_definition == "max":
        fmaxs = [force_coeff * (atoms.get_forces() ** 2).sum(axis=1).max()**(1/2) for atoms in trajectory]
        if fmaxs:
            uncertainty.append(np.std(fmaxs))
    else:
        warnings.warn_explicit(
        f"""For some reason, no forces resulted from the committee singlepoints!
        Weighted uncertainty will have no contribution from FORCE uncertainty""",
        category = UserWarning,
        filename = __file__,
        lineno = inspect.currentframe().f_lineno
        )

    if stress_coeff > 0:
        stresses = [stress_coeff * (atoms.get_stress()[0:3] ** 2).sum(axis=1).mean()**(1/2) for atoms in trajectory]
        if stresses:
            uncertainty.append(stresses)
        else:
            warnings.warn_explicit(
                    f"Weighted uncertainty will have no contribution from STRESS uncertainty",
                    category = UserWarning,
                    filename = __file__,
                    lineno = inspect.currentframe().f_lineno
                    )

    return (sum(uncertainty), len(uncertainty))



def sort_by_sizes(trajectory: List[Atoms], mode: str, np_atom_type: str = None) -> List[Atoms]:
    """
    Does:   Sorts a trajectory by densities or sizes of nanoparticles

    Parameters:
    trajectory (List[Atoms]):   Trajectory of atoms objects to be sorted
    mode (str):                 Sort by system density (if molecular) or nanoparticle/system size (if solid). Required. Defaults to 'density'.
    np_atom_type (str):         Element count to sort by. Required if mode == 'nanoparticle_size'
    """
    mode = mode.lower()
    if mode not in ("density", "nanoparticle_size"):
        warnings.warn_explicit(
                f"Mode: {mode} neither 'density' nor 'nanoparticle_size'. Defaulting to 'density'",
                filename = __file__,
                category = UserWarning,
                lineno = inspect.currentframe().f_lineno
                )
        mode = "density"

    if mode == "density":
        densities = [(np.sum(atoms.get_masses()), atoms) / atoms.cell.volume for atoms in trajectory]
        densities.sort(key = lambda x: x[0]) #sort by densities
        return [item[1] for item in densities]

    elif mode == "nanoparticle_size":
        if not np_atom_type: #because I could not figure out an automatic way to discriminate the NP from the Support (apart from possibly the Voxelization?)
            print(f"""Specify what type of atom the nanoparticle is;
            Error is from sort_by_sizes() in {argv[0]}""")
            exit(2)
        try:
            sizes = [(atoms.symbols.count(f"{np_atom_type.capitalize()}"), atoms) for atoms in trajectory]
            sizes.sort(key = lambda x: x[0]) #sort by sizes
            return [item[1] for item in sizes]

        except ValueError as v:
            print(f"Atom symbol is probably wrong. See the below error message ↓")
            print(v)



def divide_into_subsizes(trajectory: List[Atoms], divisions: int = DIVISIONS) -> List[np.ndarray]:
    """
    Does:   Divide up the given trajectory into sub-sections based on size
            assumes this trajectory is already sorted by size, using the sort_by_sizes function

    Parameters:
    trajectory (List[Atoms]):   Trajectory of size-sorted atoms objects to be classified
    divisions (int):            How many classes to create
    """
    division_size = len(trajectory) // divisions
    remainder = len(trajectory) % divisions
    
    subsizes = []
    start_index = 0

    for i in range(divisions):
        if i < remainder:
            end_index = start_index + division_size + 1
        else:
            end_index = start_index + division_size
        
        subsizes.append(trajectory[start_index:end_index])
        start_index = end_index
    
    return subsizes


def check_for_MLIP_ensemble(paths: List[str]) -> List[str]:
    """
    Checks to see if every specified MLIP for the committee ensemble exists
    Parameter:
    paths (List[str]):  Path to each MLIP
    """
    model_paths = []
    for model_path in paths:
        if not path.exists(model_path):
            warnings.warn(f"{model_path} not found! Will not use this MLIP")
        else:
            model_paths.append(model_path)
    return model_paths


def select_structures(sorted_atoms_lists: List[List[Atoms]], paths: List[str], force_mode: str) -> List[Atoms]:
    """
    Does:   Finally selects structures, considering their sizes and their uncertainties

    Parameters:
    sorted_atoms_lists (List[List[Atoms]]):     Trajectory sorted by size, then broken into blocks
    paths (List[str]):                          Absolute path to each committee MLIP
    force_mode (str):                           To evaluate force uncertainties based on 'max' force or 'mean' force
    """
    selected_structures = []
    for index, image_list in enumerate(sorted_atoms_lists):
        atoms_and_uncertainties = dict()
        for image in image_list:
            if check_system_integrity(image):
                evaluated_image = committee_singlepoint(image, model_paths = paths)
                uncertainty, calculation_type = calculate_uncertainty(evaluated_image, force_definition = force_mode)
                if uncertainty < uncertainty_threshold: #must be > for us to even consider such an image
                    continue
    #           assert calculation_type == 2, "Seems your uncertainty actually wasn't based purely on energies and forces"
                image.info["Uncertainty"] = uncertainty
                atoms_and_uncertainties[uncertainty] = image
            else:
                print("Skipping unreasonable image")
        try:
            high_uncertainty_threshold = np.percentile(np.array(list(atoms_and_uncertainties.keys())), consideration_percentile)
            bad_structures = [atoms_and_uncertainties[key] for key in atoms_and_uncertainties if key > high_uncertainty_threshold]
            shuffle(bad_structures)
            bad_structures = bad_structures[ : max_structures_to_select // len(sorted_atoms)] #make the selection from different size ranges balanced
                                                                                             #might sometimes be dangerous, like when we want mostly correct densities
                                                                                              #TODO: modify to be more intelligent
            selected_structures.extend(bad_structures)
        except Exception as e:
            print(f"Empty array; probably all low uncertainty structures:\t{e}")
            print(f"Moving on to {index + 1} block of structures")
            continue

    return selected_structures


def write_selected_images(trajectory: List[Atoms], submit_directory: str):
    """
    Does:       Writes out the selected images (if any) in a specified directory

    Parameters: 
    trajectory (List[Atoms]):   Selected Atoms objects to write
    submit_directory (str):     Absolute path to write the images to
    """
    if not path.exists(submit_directory):
        makedirs(submit_directory)
    if trajectory:
        unique_ID = time()
        write(f"{submit_directory}/selected_{unique_ID}.traj", trajectory)
        print(f"Written the {len(trajectory)} selected structures")
    else:
        print("No images selected.\nDecrease uncertainty threshold, use more diverse MLIP architectures, or be happy!")


if __name__ == "__main__":
    parser = ArgumentParser(description = "We will help in rationally selecting structures for DFT")

    #detect the ensemble of models if not given in command-line
    mlip_pattern = ["*mlip", "*mlff"]
    committee_directory = f"{getcwd()}/COMMITTEE"
    paths_to_mlips = []
    if path.exists(committee_directory):
        for pattern in mlip_pattern:
            paths_to_mlips.extend(glob(f"{committee_directory}/{pattern}"))
    parser.add_argument(
            "--model_paths", "-mp", type = str, nargs = "+", default = paths_to_mlips,
            help = "Absolute path for each MLIP to form your ensemble. In the format: x y z. Default is to search for 'COMMITTEE/{*mlip,*mlff}'"
            )

    parser.add_argument(
            "--trajectories", "-tr", type = str, nargs = "+", required = True,
            help = "Absolute path for each MD to select structures from. Required. In the format: x y z."
            )
    parser.add_argument(
            "--uncertainty_threshold", "-ut", type = float, default = UNCERTAINTY_SELECTION_THRESHOLD,
            help = f"Ignore structures of weighted_uncertainty < this. Optional. Default = {UNCERTAINTY_SELECTION_THRESHOLD}"
            )
    parser.add_argument(
            "--selection_cap", "-sc", type = int, default = MAX_STRUCTURES_TO_SELECT,
            help = f"Don't select > number of structures for DFT. Optional. Default = {MAX_STRUCTURES_TO_SELECT}"
            )
    parser.add_argument(
            "--consideration_percentile", "-percentile", type = float, default = PERCENTILE,
            help = f"Only consider structures above this percentile in uncertainties (each in its size group). Optional. Default = {PERCENTILE}"
            )
    parser.add_argument(
            "--divisions", "-dv", type = int, default = DIVISIONS,
            help = f"""How many size-based divisions of your data to make.
            Used in making sure the selected points are not too lopsided in terms of system size. Optional. Default = {DIVISIONS}"""
            )
    parser.add_argument(
            "--sizing_mode", "-sm", type = str, default = "density",
            help = "Size the system based on 'density' (molecular) or 'nanoparticle_size' (solid). Default = 'density'"
            )
    parser.add_argument("--np_atom_type", "-NP", type = str, default = None, help = "What element (if any) in your system is the nanoparticle? Optional. Default = None")
    parser.add_argument("--force_mode", "-fm", type = str, default = "mean", help = "To evaluate force uncertainties based on 'max' force or 'mean' force. Optional. Default = 'mean'")
    parser.add_argument("--submit_directory", "-dir", type = str, required = True, help = "Absolute path of directory to submit singlepoints of the selected structures. Required")
    parser.add_argument("--write_interval", "-W", type = int, required = True, help = "Interval for writing structures. Required. ¡¡SHOULD be that which was used in the MD!!")
    parser.add_argument("--timestep", "-t", type = float, required = True, help = "Timestep for running the MD (in fs). Required")
 
    args = parser.parse_args()


    #Check for the ensemble of MLIPs
    model_paths = check_for_MLIP_ensemble(args.model_paths)

    #determine the slice interval for reading the trajectory
    interval = define_selection_interval(
            md_write_interval = args.write_interval, time_interval = args.write_interval,
            timestep = args.timestep
            )

    images = []
    for trajectory in args.trajectories:
        images.append(read(trajectory, f"::{interval}"))


    force_mode = args.force_mode.lower()
    uncertainty_threshold = args.uncertainty_threshold if args.uncertainty_threshold else UNCERTAINTY_SELECTION_THRESHOLD
    max_structures_to_select = args.selection_cap if args.selection_cap else MAX_STRUCTURES_TO_SELECT
    consideration_percentile = args.consideration_percentile if args.consideration_percentile else PERCENTILE

    #sort atoms object by nanoparticle size#sort atoms object by nanoparticle size
    mode = args.sizing_mode

    divisions = args.divisions if args.divisions else DIVISIONS
    if divisions >= len(images) // 2:
        warnings.warn(f"Requested number of divisions: {divisions} too great; Decreasing")
        divisions = divisions // 4

    all_sorted_atoms = []
    for atoms in images:
        sorted_atoms = divide_into_subsizes(sort_by_sizes(atoms, mode = mode, np_atom_type = args.np_atom_type), divisions = divisions)
        all_sorted_atoms.extend(sorted_atoms)

    #Now, select structures
    selected_structures = select_structures(all_sorted_atoms, paths = model_paths, force_mode = force_mode)

    #write out the images
    write_selected_images(selected_structures, args.submit_directory)

