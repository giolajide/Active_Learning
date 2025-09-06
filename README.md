January 2025
------------
We run MD with an Allegro/NequIP MLIP, frequently to see that all is well (run_md.py and md_quality_assurance.py)
After the MD, we finally select points for DFT (points_selecter.py) using an ensemble of Allegro/NequIP MLIPs


md_quality_assurance.py is the foundational script, upon which every other script relies
run_md.py invokes it to run the MD and monitor it for equilibration (for possible early stopping) or instability
finally, points_selecter.py tried to select structures
 

Follow these steps:
__________________

1. Run the MD:
For example:

python run_md.py --temperatures 400 500 --timestep 5 --total_time 40 --write_interval 20 --friction_factor 0.02 --model_path MLIP/l02.mlip --trajectory TRAJ/reduced_group_2.traj --temp_step 40 --ramp_time 2 --start_temp 300 --check_frequency 10



2. Select points:
For example:

python points_selecter.py --model_paths MLIP/l02.mlip MLIP/42g.mlip --trajectories 0/400.0/400.0.traj 2/350.0/350.0.traj --uncertainty_threshold 0.0025 --selection_cap 30 --divisions 6 --sizing_mode nanoparticle_size --np_atom_type ag --submit_directory ${PWD}/SinglePoints --write_interval 30 -t 5


