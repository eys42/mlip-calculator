from openbabel import pybel

def convert_mol_to_xyz(filename : str):
    mol = pybel.readfile("mol", filename).__next__()
    mol.write("xyz", filename.replace(".mol", ".xyz"), overwrite=True)

def convert_xyz_to_mol(filename : str):
    mol = pybel.readfile("xyz", filename).__next__()
    mol.write("mol", filename.replace(".xyz", ".mol"), overwrite=True)