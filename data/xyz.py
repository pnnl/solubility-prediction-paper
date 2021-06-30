import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from mordred import Calculator, descriptors
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from rdkit.Chem import AllChem
import ase
from ase import Atoms
import os
import pickle
from openbabel import pybel

        
def write_pybelxyz(pybel_xyz, xyz_name, cas):
    
    natoms = int(pybel_xyz[0])
    xyz_name = xyz_name
    f = open(xyz_name,'w')
    f.write('{0:d} \n'.format( natoms ) )
    f.write('{0:s} \n'.format(cas) )
    for i in range(2, 2 + natoms):
        xn = pybel_xyz[i].split()

        f.write('{:2s} {:10.6f} {:10.6f} {:10.6f}\n'.format(xn[0], float(xn[1]), float(xn[2]), float(xn[3]) ))

    f.close()
    

def create_xyz(df):
    
    os.system("rm -r pybel_xyz")
    os.mkdir("pybel_xyz")
    save_dir = "pybel_xyz/"
    os.mkdir("pybel_xyz/with_H")
    os.mkdir("pybel_xyz/no_H")


    failed = []
    for i in tqdm(range(df.shape[0])):

        try:
            smiles = df.loc[i, 'rdkit_smiles']
            inchi = df.loc[i, 'Standard InChIKey']
            cas = df.loc[i, 'CAS Number']

            mol = pybel.readstring('smi', smiles)

            mol.addh()
            mol.make3D('mmff94') #Initial Universal Force-field optimization
            mol.localopt('mmff94') #Final Universal Force-field optimization


            xyz_name = save_dir + "with_H/"+ str(cas)+".xyz"
            pybel_xyz = mol.write("xyz").split('\n')
            write_pybelxyz(pybel_xyz, xyz_name, cas)

            mol.removeh()
            xyz_name = save_dir + "no_H/" + str(cas)+"_noH.xyz"
            pybel_xyz = mol.write("xyz").split('\n')
            write_pybelxyz(pybel_xyz, xyz_name, cas)

        except:
            failed.append([smiles, inchi])

    with open(save_dir+"pybel_failed_mols.pkl", "wb") as f:
        pickle.dump(failed, f)


