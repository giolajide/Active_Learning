"""
Controller
"""
from ase.visualize import view
from nequip.ase import NequIPCalculator
from argparse import ArgumentParser
from sys import argv, exit
import torch
import numpy as np
from ase.io import read, Trajectory, write
from os import makedirs, getcwd, path
from typing import List, Tuple
from ase import Atoms
import warnings
import inspect
import operator
from tqdm import tqdm
from random import shuffle

#custom scripts
from points_selecter import (
        define_selection_interval,
        committee_singlepoint, divide_into_subsizes,
        calculate_uncertainty, sort_by_sizes,
        DIVISIONS
        )
from md_quality_assurance import check_system_integrity

#warnings.filterwarnings("ignore")


UNCERTAINTY_SELECTION_THRESHOLD = 0.003 #I admit this is arbitrary; not at all rigorous
MAX_STRUCTURES_TO_SELECT = 50 #this, I admit, is also arbitrary
PERCENTILE = 20 #consider structures in the 80th percentile (for uncertainties) in their size groups
DIVISIONS = 3

#determine the slice interval for reading the trajectory
interval = define_selection_interval(
        md_write_interval = 20, time_interval = 3,
        timestep = 5
        )


atoms = read("../0/350.0/350.0.traj", f"::{int(interval)}")
model_paths = [
        "../MLIP/42g.mlip", "../MLIP/47g.mlip", "../MLIP/5g-zbl.mlip",
        "../MLIP/absurd.mlip", "../MLIP/l02.mlip", "../MLIP/l0.mlip"
        ]

print(len(atoms))
#sort atoms object by nanoparticle size
sorted_atoms = divide_into_subsizes(sort_by_sizes(atoms, mode = "nanoparticle_size", np_atom_type = "Ag"), divisions=DIVISIONS)

selected_structures = []
#for each image in the now derived MD trajectory, run a singlepoint with every committee member
#my uncertainties in the below will be derived only from Energy and Force uncertainty
#appended = 0
#while len(selected_structures) < MAX_STRUCTURES_TO_SELECT:
for image_list in tqdm(sorted_atoms):
    atoms_and_uncertainties = dict()
    print(len(image_list))
    for image in image_list:
        print(len(image))
        if check_system_integrity(image):
            evaluated_image = committee_singlepoint(image, model_paths = model_paths)
            uncertainty, calculation_type = calculate_uncertainty(evaluated_image, force_definition = "mean")
            if uncertainty < UNCERTAINTY_SELECTION_THRESHOLD: #must be > for us to even consider such an image to be bad
                continue
            assert calculation_type == 2, "Seems your uncertainty actually wasn't based purely on energies and forces"
            atoms_and_uncertainties[uncertainty] = image
            print(uncertainty)
        else:
            print("Image is unreasonable")
        
    try:
        high_uncertainties = np.percentile(np.array(list(atoms_and_uncertainties.keys())), PERCENTILE)
        print(atoms_and_uncertainties.keys())
        bad_structures = [atoms_and_uncertainties[key] for key in atoms_and_uncertainties if key > high_uncertainties]
        shuffle(bad_structures)
        print(f"Bad structures; {len(bad_structures)}")
        bad_structures = bad_structures[ : MAX_STRUCTURES_TO_SELECT // len(sorted_atoms)] #make the selection from different size ranges balanced
                                                                                          #this might be dangerous in some cases, like when we want mostly correct densities
                                                                                          #TODO: modify to be more intelligent; but haven't figured it out
        selected_structures.extend(bad_structures)
        print(f"Reduced bad structures: {len(bad_structures)}")
    except Exception as e:
        print("empty array; probably all low uncertainty structures")
        continue 
    

print(len(selected_structures))
print(type(selected_structures[0]))
view(selected_structures)
