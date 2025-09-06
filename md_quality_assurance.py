
"""
In this script, we develop tools to check your MD run and detect if something has gone wrong

Sept 2025: Even now, 8 months later, I think this may still be of considerable use
"""
from argparse import ArgumentParser
from sys import argv, exit
import numpy as np
from ase.io import read
from typing import List, Union
from ase import Atoms
from ase.geometry.analysis import Analysis
import warnings
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.geometry import get_duplicate_atoms
import functools

MULTIPLIER = 5
EXPANSION_THRESHOLD = 50 #percent
EQUILIBRATION_TOLERANCE = 2 #kelvin
EQUILIBRATION_WINDOW = 200
OVERLAP_CUTOFF = 0.5 #Angstrom
MIN_COORDINATION = 1 #bonds
BOND_SCALING_FACTOR = 1.08
MODE = "return"


class SimulationError(Exception):
    """Exception raised for errors that occur during simulation."""
    pass

def deprecated(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"""{function.__name__} is not the best.
            Use Demola's script instead (TODO: incorporate here)""",
            DeprecationWarning,
            stacklevel = 2
        )
        return function(*args, **kwargs)
    return wrapper


def check_for_explosion(atoms: Atoms, multiplier: float = MULTIPLIER) -> bool:
    """
    Check if the system exploded. Checks occur in this order:
        1. By the computationally cheap distance criterion
        2. By the velocities criterion
    
    Parameters:
    atoms (Atoms):      The atoms object to analyze.
    multiplier (float): Optional. Scaling constant to determine cutoffs for checking system sizes and velocities.
                        Default = 5

    Returns:
    bool: True if the object is thought to be exploded, False otherwise

    Example usage:
        from Active_Learning.md_quality_assurance import check_for_explosion
        if check_for_explosion(atoms = read('POSCAR')):
            print('System is exploded! Stopping NVT immediately')
    """
    image = atoms.copy()
    image.wrap() # IMPORTANT!!!! Ensure atoms are wrapped if possible
    x_spread = np.max(image.positions[:,0]) - np.min(image.positions[:,0])
    y_spread = np.max(image.positions[:,1]) - np.min(image.positions[:,1])
    cutoff = np.max(image.cell.lengths()) * multiplier
    print(f"Cutoff distance for checking explosion: {cutoff:.2f} Angstrom")

    print("Checking by max distance between any atom pairs")
    if (y_spread >= cutoff or x_spread >= cutoff):
        print("exploded according to max distances")
        return True

    else:
        print("Checking by comparing actual velocities against theoretical velocity distribution")
        actual_velocities = np.linalg.norm(image.get_velocities(), ord = 2, axis = 1)
        MaxwellBoltzmannDistribution(image, temperature_K = image.get_temperature(), force_temp = False)
        Stationary(image)
        maxwell_velocities = np.linalg.norm(image.get_velocities(), ord = 2, axis = 1)

        if (
                np.max(actual_velocities) > (multiplier * np.max(maxwell_velocities)) or
                np.mean(actual_velocities) > (multiplier * np.mean(maxwell_velocities))
                ):
            print("exploded according to velocities")
            return True
        else:
            warnings.warn(f"""{check_for_explosion.__name__}: We cannot definitively say that the system has NOT exploded. 
                    You might want to use check_for_floating_atoms() or analyze_neighbors_evolution() next""")
            return False


def analyze_neighbors_evolution(trajectory:  List[Atoms], threshold: float = EXPANSION_THRESHOLD) -> bool:
    """
    Determines if the system has exploded or not, using the most expensive method: tracking the 'density' of the system over time

    Parameters:
    trajectory (List[Atoms]):   The trajectory to analyze. Preferably at a constant temperature
    threshold (float):          Average number of neighbors must not increase by > this percent between two consecutive images.
                                Optional. Default = 30 (percent)

    Returns:
    bool: True if the object is thought to be exploded, False otherwise

    Example usage:
        from Active_Learning.md_quality_assurance import analyze_neighbors_evolution, track_num_neighbors
        if analyze_neighbors_evolution(atoms = read('md.traj', ':50')):
            print('System exploded! Stopping NVT immediately')
    """
    neighbors_evolution = track_num_neighbors(trajectory)
    expanding_too_quick = False
    for i in range(1, len(neighbors_evolution)):
        percent_increase = ((neighbors_evolution[i] - neighbors_evolution[i - 1]) / neighbors_evolution[i - 1]) * 100
        if percent_increase > threshold:
            expanding_too_quick = True
            break
    return expanding_too_quick


def track_num_neighbors(trajectory: List[Atoms]) -> List[float]:
    """
    Tracks how the mean number of neighbors in the system is changing over the course of the trajectory
    May be useful to see if the system is exploding
    Assumes composition is constant (in order to reuse the neighborlist)

    Parameters:
    trajectory (List[Atoms]): The trajectory to analyze. Preferably at a constant temperature

    Returns:
    List[float]: List of the average number of neighbors for each image in the trajectory

    See the analyze_neighbors_evolution() function for example usage
    """
    all_mean_neighbors = []
    cutoff = natural_cutoffs(trajectory[0])
    neighbors = NeighborList(cutoffs = cutoff, self_interaction = False, bothways = True)

    for atoms in trajectory:
        try:
            neighbors.update(atoms) #will fail if composition is changing
        except Exception as e:
            print(f"Error: {e}\n Composition is probably changing! Will skip this image")
            continue

        all_mean_neighbors.append(np.mean([
            len(np.unique(neighbors.get_neighbors(atom.index)[0])) for atom in atoms
            ])) #mean num neighbors for all atoms of each image

    return all_mean_neighbors


def check_if_equilibrated(
        trajectory: List[Atoms], target_temperature: float,
        tolerance: float = EQUILIBRATION_TOLERANCE, window: int = EQUILIBRATION_WINDOW
        ) -> bool:
    """
    Checks if the simulation is equilibrated based on the temperature fluctuation

    Parameters:
    trajectory (List[Atoms]):   The trajectory to analyze. Preferably at a constant temperature
    target_temperature (float): Temperature of the run. Presumably an NVT.
    tolerance (float):          Max st-dev of temperature over the run, for equilibration to be considered possible
                                Optional. Default = 2 (Kelvin)
    window (int):               Check for equilibration over last 'window' images of the trajectory. Default: 200 images

    Returns:
    bool: True if the run is thought to be equilibrated over the last 'window' images of the trajectory

    Example usage:
        from Active_Learning.md_quality_assurance import check_if_equilibrated
        import numpy as np
        trajectory = read('md.traj', ':')
        if check_if_equilibrated(
            trajectory = trajectory, target_temperature = np.mean([image.get_temperature() for image in trajectory]),
            window = len(trajectory)
            ):
            print('System is equilibrated! Now, you can proceed with production runs')
    """
    print("Trying to determine if the system is equilibrated, based on the temperature fluctuations")
    if len(trajectory) < window:
        print(f"""{__file__}: Trajectory has only {len(trajectory)} images,
                which is less than the min window of {window} images""")
        return False
    
    recent_temperatures = [image.get_temperature() for image in trajectory[-window : ]]
    mean_temp = np.mean(recent_temperatures)
    stdev_temp = np.std(recent_temperatures)
    
    if (np.abs(mean_temp - target_temperature) < tolerance and stdev_temp < tolerance):
        warnins.warn("""Equilibration has been determined only by checking temperature fluctuations;
        may not be sufficient in all cases""")
        return True
    else:
        print("System doesn't seem to be equilibrated")
        return False



def check_for_overlapping_atoms(atoms: Atoms, mode: str = MODE,
        cutoff: float = OVERLAP_CUTOFF) -> Union[bool, Atoms]:
    """
    Removes overlapping pairs of atoms, or simply checks for their existence

    Parameters:
    atoms (Atoms):  The atoms object to analyze.
    cutoff (float): If any interatomic distance is < 'cutoff', the involved species are flagged as 'overlapping'
                    Optional. Default = 0.65 (Angstrom)
    mode (str):     Whether to simply note if the system has floating atoms or to also remove those overlapping pairs
                    Must be 'return' to simply note, or 'remove' to also remove those pairs

    Returns:
    if mode == 'remove':
        Atoms: Returns the Atoms object with the overlapping pairs removed, if any.
    if mode == 'return':
        bool: True if it is thought that there is at least one pair of overlapping atoms; False otherwise

    Example usage:
        from Active_Learning.md_quality_assurance import check_for_overlapping_atoms
        if check_for_floating_atoms(atoms = read('Ag_Nanoparticle.POSCAR'), cutoff = 1.5, mode = 'return'):
            print('System has overlapping atoms! Quit!')
    """
    min_theoretical_equilibrium_bond_length = np.min(natural_cutoffs(atoms)) * 2
    advisable_cutoff = min_theoretical_equilibrium_bond_length / 1.6 #division by 1.6 to account for bond compression during high-temperature MD. This is admittedly not rigorous
    if cutoff > advisable_cutoff:
        warnings.warn(f"""{check_for_overlapping_atoms.__name__}:
        Requested overlap cutoff of {cutoff} Ang > advised cutoff of {advisable_cutoff:.2f} Ang
        Therefore, changing requested cutoff to advised cutoff!""")
        cutoff = advisable_cutoff

    mode = mode.lower()
    if mode not in ("return", "remove"):
        raise ValueError("mode must be 'return' or 'remove'")
    if mode == "return":
        overlapping_atoms = len(get_duplicate_atoms(
            atoms, cutoff = cutoff, delete = False)) * 2
        if overlapping_atoms:
            print("system has overlapping atoms")
            return True
        else:
            return False
    elif mode == "remove":
        return get_duplicate_atoms(atoms, cutoff = cutoff, delete = True)



def check_for_floating_atoms(
        atoms: Atoms, coord_cutoff: int = MIN_COORDINATION,
        scaling_factor: float = BOND_SCALING_FACTOR, mode: str = MODE,
        nl: NeighborList = None) -> Union[bool, Atoms]:
    """
    Removes uncoordinated atoms or atom-groups, or simply checks for their existence

    Parameters:
    atoms (Atoms):                  The atoms object to analyze.
    coord_cutoff (int):             If any atom is coordinated to less than 'coord_cutoff' atoms,
                                    flag it as 'floating'. Optional. Default = 1 (bond)
    scaling_factor (float):         Scaling factor for ASE's natural_cutoffs()
                                    Optional. Default = 1.05
    mode (str):                     Whether to simply note if the system has floating
                                    atoms or to also remove those floating atoms
                                    Must be 'return' to simply note, or 'remove' to also remove those atoms
    nl (NeighborList):              NeighborList object to reuse (if composition is constant). Optional
                                    Providing one can massively speed up evaluation
                                    on a large amount of structures

    Returns:
    if mode == 'remove':
        Atoms: Returns the Atoms object with the floating atom(s) removed, if any.
    if mode == 'return':
        bool: True if it is thought that there is at least one floating atom; False otherwise

    Example usage:
        from Active_Learning.md_quality_assurance import check_for_floating_atoms
        if check_for_floating_atoms(atoms = read('Ag_Nanoparticle.POSCAR'), coord_cutoff = 4, mode = 'return'):
            print('System has exploded! Quit!')
    """
    mode = mode.lower()
    if mode not in ("return", "remove"):
        raise ValueError("mode must be 'return' or 'remove'")
    if coord_cutoff == 1:
        warnings.warn(f"""You have gone with a default coordination cutoff of 1
        which may not proper for all systems, for example, NPs,
        where a nearly-detached atom or group may have a coordination as great as 3""")

    if not nl:
        nl = NeighborList(natural_cutoffs(
            atoms, mult = scaling_factor), self_interaction = False,
            bothways = True)
    else:
        try:
            nl.update(atoms)
        except:
            nl = NeighborList(natural_cutoffs(
                atoms, mult = scaling_factor), self_interaction = False,
                bothways = True)

    nl.update(atoms)
    bonds = nl.get_connectivity_matrix(sparse = False).sum(axis = 0)

    if mode == "return":
        if np.any(bonds < coord_cutoff):
            print("system has uncoordinated atoms")
            return True
        else:
            return False

    elif mode == "remove":
        indices_to_delete = []
        for index in range(len(atoms)):
            if bonds[index] < coord_cutoff:
                print("system has uncoordinated atoms")
                indices_to_delete.append(index)
        del atoms[indices_to_delete]
        
        if not len(atoms):
            raise SimulationError(f"""{check_for_floating_atoms.__name__}: No atoms left!
                    Your system might have exploded horribly""")

        return atoms

@deprecated
def check_autoionization(atoms, scaling_factor: float = BOND_SCALING_FACTOR,
        nl: NeighborList = None) -> bool:
    """
    Does:   Checks if water has autoionized. Checks that
            (1) each O is bound to 2 Hs ONLY
            (2) each H is bound to 1 O ONLY

    Parameters:
    atoms (Atoms):          Atoms object to be evaluated
    scaling_factor (float): Multiplier for ASE's default covalent radii.
                            Optional. Default = 1.08
    nl (NeighborList):      NeighborList object to reuse. Optional.
                            Highly encouraged for constant-composition trajectories

    Returns True if system has autoionized, else False

    Example Usage
        from Active_Learning.md_quality_assurance import check_autoionization
        atoms = read("traj.traj",":")
        nl = NeighborList(cutoffs=natural_cutoffs(atoms[0]),self_interaction=False,bothways=True)
        good_system = all([not check_autoionization(image, nl = nl) for image in atoms])
        if good_system:
            print("System did NOT go wrong!")
    """
    symbols = atoms.symbols
    if not ("H" in symbols and "O" in symbols):
        raise ValueError("This function is designed only for water-containing systems!")
    if not np.all(atoms.pbc):
        raise ValueError(f"If cell and PBC are not set, {function.__name__} will give misleading results")

    if not nl:
        nl = NeighborList(natural_cutoffs(
            atoms, mult = scaling_factor), self_interaction = False,
            bothways = True)
    else:
        try:
            nl.update(atoms)
        except:
            nl = NeighborList(natural_cutoffs(
                atoms, mult = scaling_factor), self_interaction = False,
                bothways = True)

    nl.update(atoms)
    analyzer = Analysis(atoms, nl = nl)
    O_O_bonds, H_H_bonds = analyzer.get_bonds("O", "O")[0],\
            analyzer.get_bonds("H", "H")[0] #cheap test for unwanted (O-O or H-H) bonds

    if (O_O_bonds or H_H_bonds):
        print("O-O and-or H-H bonds found!")
        return True
    else:
        O_H_bonds = analyzer.get_bonds("O", "H", unique = True)
        #for some reason, Analysis will give O-H lenghts circa 1.5A as bonds,
        #no matter how you tweak the covalent cutoffs. Now, we circumvent this:
        H_index = next((index for index, atom in enumerate(atoms) if atom.symbol == "H"), None)
        O_index = next((index for index, atom in enumerate(atoms) if atom.symbol == "O"), None)
        covalent_radii = natural_cutoffs(atoms, mult = scaling_factor)
        O_covalent_radius, H_covalent_radius = covalent_radii[O_index], covalent_radii[H_index]

        intramolecular_H_bond_indices = np.where(np.array(
            analyzer.get_values(O_H_bonds)[0]) < 1.3 * (O_covalent_radius + H_covalent_radius)
            )[0]
    
        O_H_bonds = O_H_bonds[0]
        O_H_bonds = [O_H_bonds[i] for i in intramolecular_H_bond_indices]
        Os, Hs = zip(*[(o,h) for (o, h) in O_H_bonds])
        #is any H bound to != 1 O?
        residual_H = len(O_H_bonds) - np.unique(Hs).shape[0]
        #is any O bound to != 2 H?
        residual_O = len(O_H_bonds) - (2 * np.unique(Os).shape[0])
        if (residual_O or residual_H):
            print(f"""Some connectivities are not right!
            'Residual' O: {residual_O},
            'Residual' H: {residual_H}""")
            return True
        else:
            return False


#utility function
def check_system_integrity(atoms, multiplier: float = MULTIPLIER) -> bool:
    """
    Does:   Ensures we don't consider any unreasonable structure (has overlapping atoms and-or uncoordinated atoms)
            Combines several of the above functions into one easy-to-use, portable format

    Parameters:
    atoms (Atoms):      Atoms object to be evaluated
    multiplier (float): Multiplier for checking distances and velocities. Optional. Default = 1.5

    Returns True if the system is thought to be sane, else False
    """
    print("Running checks . . .")
    if (
            check_for_explosion(atoms, multiplier = multiplier)
            or check_for_overlapping_atoms(atoms)
            or check_for_floating_atoms(atoms)
            ):
        print("MD has overlapping atoms and-or has exploded")
        return False
    else:
        return True


