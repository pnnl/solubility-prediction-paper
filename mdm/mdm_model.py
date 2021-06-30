import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import config

act = {0:'relu', 1:'selu', 2:'sigmoid'}
args={'a1': 2, 'a2': 0, 'a3': 1, 'a4': 1, 'a5': 0, 'bs': 1, 'd1': 0.10696194799818459, 'd2': 0.6033824611348487,\
      'd3': 0.7388531806558837, 'd4': 0.9943053700072028, 'd5': 0.016358259737496605, 'h1': 128.0, 'h2': 576.0,\
      'h3': 448.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}

def create_model(input_shape):        
    model = Sequential()
    model.add(Dense(int(args['h1']), input_shape = (input_shape,) ))
    model.add(Activation( act[args['a1']] ))
    model.add(Dropout( args['d1'] ))
    model.add(Dense(  int(args['h2'])  ))
    model.add(Activation( act[args['a2']] ))
    model.add(Dropout( args['d2'] ))

    model.add(Dense(1, activation = 'linear'))
    rmsprop = keras.optimizers.RMSprop( lr = config.lr )
    opt = rmsprop

    model.compile(loss='mean_squared_error', optimizer=opt)
    
    return model