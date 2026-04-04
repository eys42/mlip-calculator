from ase import Atoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.io import mol
from molecule import Molecule
from convert import convert_xyz_to_mol
from torch import cuda


def convert_file_to_molecule(filename : str, model : str = 'uma-m-1p1', charge : int = 0, spin : int = 1) -> Molecule:
    if filename.endswith(".xyz"):
        convert_xyz_to_mol(filename)
        print(f'Copied structure in {filename} to {filename.replace(".xyz", ".mol")}.')
        filename = filename.replace('.xyz', '.mol')
        
    if model not in ['uma-s-1p2', 'uma-s-1p1', 'uma-m-1p1']:
        raise ValueError(f'Model {model} not found.')
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Using device: {device} for inference with model {model}.')
    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    with open(filename, 'r') as file:
        base_fname = filename.split('.')[0]
        structure : Atoms = mol.read_mol(file)
        molecule : Molecule = Molecule(structure, base_fname=base_fname, charge=charge, spin=spin)
        molecule.set_calc(FAIRChemCalculator(predictor, task_name='omol'), model=model)
        return molecule

def optimize_min(molecule : Molecule, fmax : float = 0.01, max_steps : int = 500):
    print((f'Starting optimization of {molecule.get_base_fname()} (charge={molecule.get_charge()}, '
          f'spin={molecule.get_spin()}) with model {molecule.get_model()}.\nMax steps: {max_steps}, max force threshold: {fmax} eV/Å.'))
    molecule.optimize_min(max_steps=max_steps, fmax=fmax, trajectory_file=f'{molecule.get_base_fname()}_opt_{molecule.get_model()}.traj')
    print(f'Optimization converged in {molecule.get_time_to_optimize():.2f} seconds.')
    molecule.save_to_file(f'{molecule.get_base_fname()}_optimized_{molecule.get_model()}.xyz', format = 'xyz')
    print(f'Optimized structure saved to {molecule.get_base_fname()}_optimized_{molecule.get_model()}.xyz')

def freq_and_thermo(molecule : Molecule, geometry : str = 'nonlinear', symmetrynumber : int = 1, temperature : float = 298.15, pressure : float = 101325.0):
    print(f'Calculating vibrational frequencies for {molecule.get_base_fname()} with model {molecule.get_model()}.')
    molecule.calculate_vibrational_frequencies()
    print(f'Vibrational frequencies calculated in {molecule.get_time_to_vibrational_frequencies():.2f} seconds.')
    print(f'Calculating thermochemical properties for {molecule.get_base_fname()} with model {molecule.get_model()} at T={temperature} K and P={pressure} Pa.')
    
    molecule.calculate_thermochemistry(geometry=geometry, symmetrynumber=symmetrynumber, temperature=temperature, pressure=pressure)