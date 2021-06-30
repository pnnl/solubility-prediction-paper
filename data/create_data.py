import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from mordred import Calculator, descriptors
from tqdm import tqdm
from mol_des_utils import create_momentf
import mordred
import pickle
from xyz import create_xyz
import datetime

def get_xyz(path):
    f = open(path, 'r')
    f = f.readlines()
    atoms = f[2:]
    
    natoms = int(f[0].strip('\n'))
    
    
    atom_z = [atoms[i].split()[0] for i in range(natoms)]
    pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms)]
#     pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms) if atom_z[i] !='H']
#     atom_z = [atom_z[i] for i in range(natoms) if atom_z[i] !='H']
    
    return atom_z, np.array(pos)

def check_for_nan_coords(df_3d, xyz_dir):
    
    nans =[]
    for i in tqdm(range(df_3d.shape[0])):

        fname = df_3d.loc[i, 'cas']
    #         fname = str(fname)
        path = xyz_dir + fname + ".xyz"
    #         print(path)
        try:
            anum, pos = get_xyz(path)

            if len(np.where(np.isnan(pos.ravel()))[0]) !=0:
                nans.append(fname)

        except:
            nans.append(fname)
    #     break
    return nans

def find_missing(df_3d, features):
    
    missing_features = []
    for f in features:    
        for i in df_3d.loc[:, f].values:
            if isinstance(i , mordred.error.Missing):
                missing_features.append(f)
    #             print(i)
                break
    return missing_features

def get_float_features(dft):
    
    floats=[]
    for i in dft.columns:
        try:
            dft[i].astype(float)
            floats.append(i)
        except:
            pass
    return floats

# location of the xyz coordinates
xyz_dir = './pybel_xyz/with_H/'

# common fragments
with open("./fragments.dat", "rb") as f:
    fragments = pickle.load(f)

# load the data downloaded from  
# https://figshare.com/s/6258a546a27a2373bf2a/articles/14558808
df = pd.read_csv("./data.csv", encoding='iso-8859-1')

# convert smiles to canonical format
df['rdkit_smiles'] = np.nan
for i in tqdm(range(df.shape[0])):
    try:
        df.loc[i,'rdkit_smiles'] = Chem.MolToSmiles(Chem.MolFromSmiles(df.loc[i, 'SMILES']))
    except:
        pass
    

df['log_sol'] = np.log10(1e-3*df['Experimental Solubility in Water']/df['Molar Mass'])
df.reset_index(drop=True, inplace=True)
dups = df[df.duplicated(subset=['rdkit_smiles'])]

if dups.shape[0] !=0:
    print("Warning: There are duplicate smiles!")

else:    
    # create pybel xyz data
    create_xyz(df)

    # calculate mordred features
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(i) for i in df.rdkit_smiles.values]
    df_3d = calc.pandas(mols)
    features = df_3d.columns.tolist()

    # find features containing missing values
    # such columns are removed
    missing_features = find_missing(df_3d, features)      
    newf = list(set(df_3d.columns.values).difference(list(set(missing_features))))
    df_3d = df_3d.loc[:, newf]
    
    # only use features that can be converted to floats
    floats = get_float_features(df_3d)
    df_3d = df_3d.loc[:, floats]
    
    # drop other nan columns
    df_3d = df_3d.astype(float)
    df_3d.dropna('columns', inplace=True)
    df_3d.reset_index(drop=True, inplace=True)

    # add the important data
    df_3d.loc[:, 'smiles'] = df.rdkit_smiles
    df_3d.loc[:, 'log_sol'] = df.log_sol
    df_3d.loc[:, 'cas'] = df['CAS Number']
    df_3d.loc[:, 'ref'] = df['Experiment Reference']
    df_3d.loc[:, 'temp'] = df['Temperature']
    df_3d.loc[:, 'inchi'] = df['Standard InChIKey']

    df_3d = create_momentf(df_3d, xyz_dir, 'cas', fragments)


    # check whether there are nan coordinates or xyz coordinates are unavailable
    nans = check_for_nan_coords(df_3d, xyz_dir)
    df_3d = df_3d[~df_3d.cas.isin(nans)]
    df_3d.reset_index(drop=True, inplace=True)


    # split the data into train/valid/test sets
    df_3d.loc[:,"sol_id"] = np.nan
    df_3d.loc[df_3d.log_sol<=-8,"sol_id"] = 0
    df_3d.loc[(df_3d.log_sol>-8) & (df_3d.log_sol<=-6),"sol_id"] = 1
    df_3d.loc[(df_3d.log_sol>-6) & (df_3d.log_sol<=-4),"sol_id"] = 2
    df_3d.loc[(df_3d.log_sol>-4) & (df_3d.log_sol<=-2),"sol_id"] = 3
    df_3d.loc[(df_3d.log_sol>-2) & (df_3d.log_sol<=0),"sol_id"] = 4
    df_3d.loc[df_3d.log_sol>0 ,"sol_id"] = 5

    train, test = train_test_split(df_3d, stratify = df_3d.sol_id.values, test_size=0.15)
    val, test = train_test_split(test, stratify = test.sol_id.values, test_size=0.5)

    train.drop(['sol_id'], axis=1, inplace=True)
    test.drop(['sol_id'], axis=1, inplace=True)
    val.drop(['sol_id'], axis=1, inplace=True)

    train.to_csv("./train.csv", index=False)
    test.to_csv("./test.csv", index=False)
    val.to_csv("./val.csv", index=False)

    print(f"data creation completed at {datetime.datetime.now()}")
