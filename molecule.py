from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
from time import time
# pyright: reportArgumentType=false

class Molecule():
    """
    A class representing a molecule, wrapping from ASE's Atoms class.
    Attributes:
        _time_to_optimize (float): Time taken to optimize the molecule.
        atoms (Atoms): The ASE Atoms object representing the molecule.
        _is_optimized (bool): Whether the molecule has been optimized.
    """
    _time_to_optimize : float
    _time_to_vibrational_frequencies : float
    _atoms : Atoms
    _vibrations : Vibrations
    _vibrationsdata : VibrationsData
    _is_optimized : bool = False
    _has_vibrational_frequencies : bool = False
    _base_fname : str
    _model : str
    _thermo : IdealGasThermo
    _gibbs_energy : float
    
    def __init__(self, atoms : Atoms, base_fname : str, charge : int = 0, spin : int = 1):
        """
        Instantiates a Molecule object with the given ASE Atoms object.
        :param atoms: An ASE Atoms object representing the molecule.
        :param base_fname: The base filename for the molecule.  
        :param spin: The spin multiplicity of the molecule.
        :param charge: The total charge of the molecule.
        """
        self._atoms = atoms
        self._atoms.info.update({"spin": spin, "charge": charge})
        self._base_fname = base_fname

    def get_charge(self) -> int:
        """
        Returns the total charge of the molecule.
        """
        return self._atoms.info['charge']
    
    def get_spin(self) -> int:
        """
        Returns the spin multiplicity of the molecule.
        """
        return self._atoms.info['spin']
    
    def get_base_fname(self) -> str:
        """
        Returns the base filename of the molecule.
        """
        return self._base_fname
    
    def get_model(self) -> str:
        """
        Returns the model used for calculations on the molecule.
        """
        return self._model

    def set_calc(self, calc, model : str):
        """
        Sets the calculator for the molecule.
        :param calc: The calculator to set for the molecule.
        :param model: The model to use for the calculation.
        """
        self._atoms.calc = calc
        self._model = model

    def optimize_min(self, optimizer = BFGS, max_steps : int = 100, fmax : float = 0.01, restart_file : str | None = None, trajectory_file : str | None = None):
        """
        Optimizes the molecular structure to a local minimum using the specified 
        optimization method and sets the time taken for optimization.
        :param optimizer: The optimization algorithm to use (default is BFGS).
        :param max_steps: Maximum number of optimization steps (default is 100).
        :param fmax: Maximum force threshold for convergence (default is 0.01 eV/Å).
        :param restart_file: File to restart the optimization from (default is None).
        :param trajectory_file: File to save the optimization trajectory (default is None).
        """
        opt = optimizer(self._atoms, restart = restart_file, trajectory = trajectory_file)
        t1 = time()
        opt.run(fmax = fmax, steps = max_steps)
        self._time_to_optimize = time() - t1
        self._is_optimized = True

    def get_info(self) -> str:
        """
        Returns a string with information about the molecule.
        """
        return "null"
    
    def get_time_to_optimize(self) -> float:
        """
        Returns the time taken to optimize the molecule.
        """
        return self._time_to_optimize
        
    def save_to_file(self, filename : str, format : str = 'xyz'):
        """
        Saves the molecule as a file of the specified format.
        :param filename: The name of the file to save the molecule to.
        :param format: The format of the file to save (default is 'xyz').
        """
        self._atoms.write(filename, format = format)

    def get_atoms(self) -> Atoms:
        """
        Returns the ASE Atoms object representing the molecule.
        """
        return self._atoms
    
    def calculate_vibrational_frequencies(self):
        """
        Calculates the vibrational frequencies of the molecule.
        """
        if not self._is_optimized:
            raise ValueError("Molecule must be optimized before calculating vibrational frequencies.")
        self._vibrations : Vibrations = Vibrations(self._atoms)
        t1 = time()
        self._vibrations.run()
        self._time_to_vibrational_frequencies = time() - t1
        self._vibrationsdata = self._vibrations.get_vibrations()
        print(self._vibrationsdata.tabulate())
        self._has_vibrational_frequencies = True

    def get_time_to_vibrational_frequencies(self) -> float:
        """
        Returns the time taken to calculate vibrational frequencies.
        """
        return self._time_to_vibrational_frequencies

    def calculate_thermochemistry(self, geometry : str = 'nonlinear', symmetrynumber : int = 1, temperature : float = 298.15, pressure : float = 101325.0):
        """
        Calculates the thermochemical properties of the molecule using the ideal gas approximation.
        :param temperature: The temperature in Kelvin (default is 298.15 K).
        :param pressure: The pressure in Pascals (default is 101325 Pa).
        """
        if not self._has_vibrational_frequencies:
            raise ValueError("Vibrational frequencies must be calculated before calculating thermochemistry.")
        vib_energies = self._vibrations.get_energies()
        self._thermo = IdealGasThermo(
            vib_energies=vib_energies,
            potentialenergy=self._atoms.get_potential_energy(),
            atoms=self._atoms,
            symmetrynumber=symmetrynumber,
            spin=self.get_spin(),
            geometry=geometry
        )
        self._gibbs_energy = self._thermo.get_gibbs_energy(temperature, pressure)

    def get_thermo(self) -> IdealGasThermo:
        """
        Returns the IdealGasThermo object containing the thermochemical properties of the molecule.
        """
        return self._thermo