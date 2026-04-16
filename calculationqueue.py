import pandas as pd
import sys
from calculate import convert_file_to_molecule, optimize_min, freq_and_thermo

class CalculationQueue:
    def __init__(self, filename):
        self.filename = filename
        self.base_fname = filename.split('.')[0]
        self.df = pd.read_excel(filename)
        self.columns = self.df.columns.tolist()
        cols = ['logfile', 'filename', 'model', 'charge', 'spin', 'fmax', 'max_steps', 'temperature', 'pressure', 'geometry', 'symmetrynumber']
        for col in cols:
            if col not in self.columns:
                raise ValueError(f'Missing required column: {col} in {filename}.')
        self.results_df = None
            
        

    def _calculate(self, row, index):
        try:
            print(f"Processing '{row['filename']}' with model {row['model']} (charge={row['charge']}, spin={row['spin']}): index {index}.")
            with open(row['logfile'], 'w') as logfile:
                sys.stdout = logfile
                mol_obj = convert_file_to_molecule(row['filename'], model=row['model'], charge=row['charge'], spin=row['spin'])
                optimize_min(mol_obj, fmax=row['fmax'], max_steps=row['max_steps'])
                freq_and_thermo(mol_obj, temperature=row['temperature'], pressure=row['pressure'], geometry=row['geometry'], symmetrynumber=row['symmetrynumber'])
                sys.stdout = sys.__stdout__
            thermo = mol_obj.get_thermo()
            row['E_pot'] = mol_obj.get_atoms().get_potential_energy()
            row['E_ZPE'] = thermo.get_ZPE_correction()
            row['Cv_trans (0->T)'] = thermo.get_ideal_translational_energy(row['temperature'])
            row['Cv_rot (0->T)'] = thermo.get_ideal_rotational_energy(row['geometry'],row['temperature'])
            row['Cv_vib (0->T)'] = thermo.get_vib_energy_contribution(row['temperature'])
            row['U'] = thermo.get_internal_energy(temperature=row['temperature'])
            row['H'] = thermo.get_enthalpy(temperature=row['temperature'])
            S, S_dict = thermo.get_ideal_entropy(row['temperature'],
                                                 translation = True,
                                                 vibration = True,
                                                 rotation = True,
                                                 geometry = row['geometry'],
                                                 pressure = row['pressure'],
                                                 electronic = True,
                                                 symmetrynumber=row['symmetrynumber'])
            row['S_trans'] = S_dict['S_t']
            row['S_rot'] = S_dict['S_r']
            row['S_vib'] = S_dict['S_v']
            row['S_elec'] = S_dict['S_e']
            row['S'] = S
            row['G'] = thermo.get_gibbs_energy(temperature=row['temperature'], pressure=row['pressure'])
            if self.results_df is None:
                self.results_df = pd.DataFrame([row])
            else:
                self.results_df = pd.concat([self.results_df, pd.DataFrame([row])], ignore_index=True)
            print(f"Finished processing '{row['filename']}'. Results logged to '{row['logfile']}'.")
        except Exception as e:
            print(f"Error processing '{row['filename']}': {e}")

    def run(self):
        for index, row in self.df.iterrows():
            self._calculate(row, index)
        print("All calculations completed.")
        self.results_df.to_excel(f'{self.base_fname}_results.xlsx', index=False)
        print(f"Results saved to '{self.base_fname}_results.xlsx'.")