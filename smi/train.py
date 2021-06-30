import keras
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras.preprocessing import sequence
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from rdkit import Chem
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import json
import os
import smi_utils
import smi_model
import config
import datetime

def run():
    
    train = pd.read_csv(config.data_dir+"train.csv")
    val = pd.read_csv(config.data_dir+"val.csv")
    test = pd.read_csv(config.data_dir+"test.csv")

    trainx, valx, testx = train, val, test
    smiles_train = list(trainx.smiles.values.ravel())
    smiles_val = list(valx.smiles.values.ravel())
    smiles_test = list(testx.smiles.values.ravel())
    smiles = smiles_train + smiles_val + smiles_test

    x_train, x_val, x_test, y_train, y_val, y_test, max_features, maxlen, tokenizer = smi_utils.get_data(trainx = trainx,
                                                                                        valx   = valx,
                                                                                        testx  = testx,
                                                                                       all_smiles = smiles)

    model = smi_model.create_model(max_features = max_features, maxlen = maxlen)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = config.patience)
    mc = ModelCheckpoint( config.best_model, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    result = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config.max_epochs,
                       batch_size = config.batch_size, verbose = True, callbacks=[es,mc])

    print(f"training completed at {datetime.datetime.now()}")
    
if __name__ == "__main__":
    run()
