from time import time
from dotenv import load_dotenv

load_dotenv()

from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit('uma-m-1p1')

from ase.build import molecule
from ase.io import mol



"""
#  singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")
t2 = time()

#  triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")
"""

t1 = time()

energy_monomer = 0
energy_dimer = 0

with open("1-Br-quintet.mol", "r") as file:
    structure = mol.read_mol(file)
    structure.info.update({"spin": 5, "charge": 0})
    structure.calc = FAIRChemCalculator(predictor, task_name="omol")
    energy_monomer = structure.get_potential_energy()
    print(structure.get_potential_energy())



with open("2-singlet.mol", "r") as file:
    structure = mol.read_mol(file)
    structure.info.update({"spin": 1, "charge": 0})
    structure.calc = FAIRChemCalculator(predictor, task_name="omol")
    energy_dimer = structure.get_potential_energy()
    print(structure.get_potential_energy())

t2 = time()

print(energy_dimer - 2 * energy_monomer)

print(f"Time taken: {t2 - t1:.2f} seconds")