import time
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
import ase
from rdkit import Chem
from ase import Atoms
import schnetpack
import torch.nn.functional as F
from torch.optim import Adam
import schnetpack.atomistic as atm
import schnetpack.representation as rep
import torch.nn as nn
import pickle
import sch_utils
import config
import schnetpack as spk
from schnetpack.datasets import AtomsData
import sch_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import datetime

def run():
    
    train = pd.read_csv(config.data_dir+"train.csv")
    val = pd.read_csv(config.data_dir+"val.csv")
    test = pd.read_csv(config.data_dir+"test.csv")

    # create dbs
    if config.create_data:
        sch_utils.create_db(train, "schtrain", config.data_dir, 'cas', config.xyz_dir )
        sch_utils.create_db(val, "schval", config.data_dir, 'cas', config.xyz_dir )
        sch_utils.create_db(test, "schtest", config.data_dir, 'cas', config.xyz_dir )

    
    train = AtomsData(config.data_dir+'schtrain.db', available_properties=['sol'])
    val = AtomsData(config.data_dir+'schval.db', available_properties=['sol'])
    test = AtomsData(config.data_dir+'schtest.db', available_properties=['sol'])

    train_loader = spk.AtomsLoader(train, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = spk.AtomsLoader(val, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = spk.AtomsLoader(test, batch_size=config.batch_size, shuffle=False, drop_last=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epochs = config.max_epochs
    model = sch_model.model
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterian = nn.MSELoss()


    early_stopping = sch_utils.EarlyStopping(patience=config.patience, verbose=True, chkpoint_name = config.best_model)
    os.system(f"rm {config.best_model}")
    hist = {"train_mse":[], "val_mse":[]}
    for i in range(max_epochs):
        model.train();
        for batch in train_loader:


            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            result = model(batch)
            loss = criterian(result['sol'], batch['sol'])
            loss.backward()
            optimizer.step()

        tr_true, tr_pred = sch_utils.valid_fn(train_loader, model, device)
        tr_loss = mean_squared_error(y_pred=tr_pred, y_true=tr_true)

        val_true, val_pred = sch_utils.valid_fn(val_loader, model, device)
        validation_loss = mean_squared_error(y_pred=val_pred, y_true=val_true)

        hist['train_mse'].append(tr_loss)
        hist['val_mse'].append(validation_loss)


        print(validation_loss)
        early_stopping(validation_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"training completed at {datetime.datetime.now()}")
    
    
if __name__ == "__main__":
    run()    
        
    
