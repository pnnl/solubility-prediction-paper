import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from rdkit.Chem import Draw
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pandas as pd
from random import randrange
import itertools
import random
import os
from pickle import dump, load
from sklearn.metrics import mean_absolute_error
import pickle
import gzip, pickle
from torch_geometric.data import DataLoader
import gnn_utils
import gnn_model
from gnn_model import GNN
import config
import datetime

def run():
    
    # create data
    gnn_utils.create_data()

    with gzip.open(f"{config.data_dir}train.pkl.gz", "rb") as f:
        train_X = pickle.load(f)
    with gzip.open(f"{config.data_dir}val.pkl.gz", "rb") as f:
        val_X = pickle.load(f)
    with gzip.open(f"{config.data_dir}test.pkl.gz", "rb") as f:
        test_X = pickle.load(f)


    # define model
    n_features = config.n_features # number of node features
    bs = config.bs

    train_loader = DataLoader(train_X, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_X, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_X, batch_size=bs, shuffle=False, drop_last=False)

    train_loader_no_shuffle = DataLoader(train_X, batch_size = bs, shuffle=False, drop_last=False)
    val_loader_no_shuffle = DataLoader(val_X, batch_size = bs, shuffle=False, drop_last=False)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(n_features = n_features).to(device)
    adam = torch.optim.Adam(model.parameters(), lr = config.lr )
    optimizer = adam
    early_stopping = gnn_utils.EarlyStopping(patience = config.patience, verbose=True, chkpoint_name = config.best_model)
    criterion = nn.MSELoss()
    n_epochs = config.max_epochs



    # train the model
    hist = {"train_rmse":[], "val_rmse":[]}
    for epoch in range(0, n_epochs):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.reshape(-1,)

            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()


        train_rmse = gnn_utils.test_fn(train_loader, model, device)
        val_rmse = gnn_utils.test_fn(val_loader, model, device)
        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)
        print(f'Epoch: {epoch}, Train_rmse: {train_rmse:.3}, Val_rmse: {val_rmse:.3}')

    print(f"training completed at {datetime.datetime.now()}")
    
    
if __name__ == "__main__":
    run()
