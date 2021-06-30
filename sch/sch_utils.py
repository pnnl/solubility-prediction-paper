import torch
import numpy as np
from rdkit import Chem
import multiprocessing
import logging
from torch_geometric.data import Data
import pandas as pd
import ase
from rdkit import Chem
from ase import Atoms
import ase.visualize
import numpy as npz
import schnetpack
import torch.nn.functional as F
from torch.optim import Adam
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import AtomsData
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import config
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, chkpoint_name='best_model.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.chkpoint_name = chkpoint_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.chkpoint_name)
        self.val_loss_min = val_loss
        
        
        
def get_xyz(path):
    f = open(path, 'r')
    f = f.readlines()
    atoms = f[2:]
    
    natoms = int(f[0].strip('\n'))
    
    
    atom_z = [atoms[i].split()[0] for i in range(natoms)]
    pos = [[float(atoms[i].split()[1:][j]) for j in range(3)] for i in range(natoms)]
    
    return atom_z, np.array(pos)


def create_db(dfx, db_name, db_dir, id_col, xyz_dir):
    
    atoms_list = []
    properties_list = []
    for i in tqdm(range(dfx.shape[0])):

        cas = dfx[ id_col].values[i]
        cas = str(cas)
        path = xyz_dir + cas + "_noH.xyz"

        Z_atoms, positions_new= get_xyz(path)
        
        atoms = Atoms(symbols = Z_atoms, positions = positions_new)

        atoms_list.append(atoms)
        solx = dfx.log_sol.values[i]
        solx = solx.astype('float32').reshape(1,) 
        
        smiles = dfx.smiles.values[i]
        smiles = np.array(smiles).reshape(1,) 
        
        properties_list.append({'sol': solx})

    db_name = db_name+".db"
    new_db = AtomsData(db_dir+"/"+db_name, available_properties=['sol'])
    new_db.add_systems(atoms_list, properties_list)
    
    
    
def valid_fn(loader, model, device):

    model.eval()

    # predict molecular properties
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)
            targets += batch['sol'].cpu().squeeze().tolist()
            predictions += result['sol'].cpu().squeeze().tolist()

    predictions = np.array(predictions)
    targets = np.array(targets)
    
    return targets, predictions
        
    
def get_results(db_name, loader, model, device):

    print(f"{db_name} results")
    
    model.eval()

    # predict molecular properties
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)
            targets += batch['sol'].cpu().squeeze().tolist()
            predictions += result['sol'].cpu().squeeze().tolist()
            
    pred = np.array(predictions)
    y = np.array(targets)
    
    r2 = r2_score(y_pred = pred, y_true = y)
    rmse = mean_squared_error(y_pred = pred, y_true = y)**.5
    sp = spearmanr(pred, y)[0]
    mae = mean_absolute_error(y_pred = pred, y_true = y)

    print("r2: {0:.4f}".format(r2) )
    print("sp: {0:.4f}".format(sp) )
    print("rmse: {0:.4f}".format(rmse) )
    print("mae: {0:.4f}".format(mae) )

    plt.plot( y, pred, 'o')
    plt.xlabel("True (logS)", fontsize=15, fontweight='bold');
    plt.ylabel("Predicted (logS)", fontsize=15, fontweight='bold');
    plt.show()
    