from dotenv import load_dotenv
from sys import stdout

from calculate import convert_file_to_molecule, optimize_min, freq_and_thermo

load_dotenv()

with open('logfile.txt', 'w') as logfile:
    stdout = logfile
    mol_obj = convert_file_to_molecule('insalencl_optimized_uma-m-1p1.xyz', model='uma-m-1p1', charge=0, spin=1)
    optimize_min(mol_obj, fmax=0.01, max_steps=500)
    freq_and_thermo(mol_obj, temperature=298.15, pressure=101325.0, geometry='nonlinear', symmetrynumber=1)
    logfile.close()