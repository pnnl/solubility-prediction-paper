from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import config

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


def get_data(trainx, valx, testx, all_smiles):
    smiles_train = list(trainx.smiles.values.ravel())
    smiles_val = list(valx.smiles.values.ravel())
    smiles_test = list(testx.smiles.values.ravel())

    smiles = all_smiles
    
    chars = sorted(list(set(''.join(list(smiles)))))
    int2char = dict(enumerate(chars,1))
    char2int = {char:ind for ind,char in int2char.items()}

    maxlen = max([len(x) for x in smiles])

    def tokenize(x):
        x = list(x)
        x = [char2int[c] for c in x] + [0]*(maxlen - len(x))
        return(x)

    X_train = np.array([tokenize(sm) for sm in smiles_train])
    X_val = np.array([tokenize(sm) for sm in smiles_val])
    X_test = np.array([tokenize(sm) for sm in smiles_test])

    smiles_len = (X_train > 0.0).sum(axis=1)
    max_features = len(chars) + 1 
    
    os.makedirs("input", exist_ok=True)
    np.savetxt("./input/x_train.txt", X_train)
    np.savetxt("./input/x_val.txt", X_val)
    np.savetxt("./input/x_test.txt", X_test)
    np.savetxt("./input/y_train.txt", trainx.log_sol.values)
    np.savetxt("./input/y_val.txt", valx.log_sol.values)
    np.savetxt("./input/y_test.txt", testx.log_sol.values)
    
    
    return X_train, X_val, X_test, trainx.log_sol.values, valx.log_sol.values, testx.log_sol.values, \
            max_features, maxlen, tokenize 


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
