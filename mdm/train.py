import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.initializers import random_normal, random_uniform
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from sklearn.model_selection import KFold
from rdkit import Chem
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import json
import mdm_model
import config
import mdm_utils
import datetime

def run():
    
    to_remove = [ 'cas', 'ref', 'temp','inchi'] 

    train = pd.read_csv(config.data_dir+"train.csv")
    val = pd.read_csv(config.data_dir+"val.csv")
    test = pd.read_csv(config.data_dir+"test.csv")


    train = train.drop(to_remove, axis=1)
    val = val.drop(to_remove, axis=1)
    test = test.drop(to_remove, axis=1)
    mdm_utils.check_duplicates(train,val,test)

    trainx = train
    valx = val
    testx = test

    to_drop = ['log_sol', 'smiles']

    x_train,y_train, x_test, y_test, x_val, y_val, sc = mdm_utils.get_transformed_data(train   = trainx, 
                                                                             val     = valx, 
                                                                             test    = testx, 
                                                                             to_drop = to_drop, 
                                                                             y       = "log_sol")


    model = mdm_model.create_model(x_train.shape[1])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config.patience)
    os.system(f"rm {config.best_model}")
    mc = ModelCheckpoint(f'{config.best_model}', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    result = model.fit(x_train, y_train, batch_size = config.batch_size, epochs = config.max_epochs,
              verbose = 2, validation_data = (x_val,y_val), callbacks = [es,mc])


    print(f"training completed at {datetime.datetime.now()}")

    
if __name__ == "__main__":
    run()
