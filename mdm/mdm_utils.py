import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def get_transformed_data(train,val,test, to_drop, y):

    x_train = train.drop(to_drop, axis=1)
    x_val = val.drop(to_drop , axis=1)
    x_test = test.drop(to_drop , axis=1)

    y_train = train[y].values
    y_val = val[y].values
    y_test = test[y].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    
    os.makedirs("input", exist_ok=True)
    np.savetxt("./input/x_train.txt", x_train)
    np.savetxt("./input/x_val.txt", x_val)
    np.savetxt("./input/x_test.txt", x_test)
    np.savetxt("./input/y_train.txt", y_train)
    np.savetxt("./input/y_val.txt", y_val)
    np.savetxt("./input/y_test.txt", y_test)
    
    return x_train,y_train, x_test, y_test, x_val, y_val, scaler

def val_results(x_valx, y_valx, lc_name, modelx):

    model = modelx
    
    print("val scores")
    pred = model.predict(x_valx).reshape(-1,)
    print(r2_score(y_pred = pred, y_true = y_valx))
    print(mean_squared_error(y_pred = pred, y_true = y_valx)**.5)
    print(spearmanr(pred, y_valx))


def test_results(x_testx, y_testx, acc_plot_name, modelx ):

    model = modelx
    
    print("test scores")
    pred = model.predict(x_testx).reshape(-1,)
    print(r2_score(y_pred=pred, y_true=y_testx))
    print(mean_squared_error(y_pred=pred, y_true=y_testx)**.5)
    print(spearmanr(pred, y_testx))

    return pred


def check_duplicates(train,val,test):
    print("checking for duplicates")
    if len(list(set(train.smiles.values).intersection(set(test.smiles.values)) )) == 0:
        print("no duplicates in train and test")
    if len(list(set(train.smiles.values).intersection(set(val.smiles.values)) )) == 0:
        print("no duplicates in train and valid")
    if len(list( set(test.smiles.values).intersection(set(val.smiles.values)) )) == 0:
        print("no duplicates in test and valid")
    print(" ")
    print(f"train set size = {train.shape}, unique smiles in the train set = {len(set(train.smiles.values))}")
    print(f"train set size = {val.shape}, unique smiles in the train set = {len(set(val.smiles.values))}")
    print(f"train set size = {test.shape}, unique smiles in the train set = {len(set(test.smiles.values))}")
    print(" ")

    
def get_results(db_name,X, y, model):
    
    print(f"{db_name} results")
    pred = model.predict(X).ravel()

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