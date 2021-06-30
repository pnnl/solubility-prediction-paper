import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
import config

args = {'a1': 2, 'a3': 0, 'a4': 0, 'a5': 0, 'bs': 2, 'd1': 0.08607118576024131, 'd2': 0.4730059911045743, \
        'd3': 0.186637772607526, 'd4': 0.27122468227787655, 'd5': 0.15564916131523265, \
        'ed': 960.0, 'h1': 704.0, 'h3': 640.0, 'h4': 704.0, 'h5': 576.0, 'lr': 0, 'ls1': 256.0, 'nfc': 0, 'opt': 2}

act = {0:'relu', 1:'selu', 2:'sigmoid'}
def create_model(max_features, maxlen):
    embedding_dim = 64
    model = Sequential()
    model.add(Embedding(max_features, int(args['ed']), input_length=maxlen))
    model.add(LSTM( int(args['ls1']), return_sequences=True))
    model.add(Dropout( args['d1'] ) )
    model.add(Bidirectional(LSTM( int(args['ls1']) )))
    model.add(Dropout( args['d2'] ))
    model.add(Dense(int( args['h1'] )))
    model.add(Activation( act[args['a1']] ))

    model.add(Dense( int(args['h3']) ))
    model.add(Activation( act[args['a3']] ))
    model.add(Dropout( args['d3'] ))


    model.add(Dense(1, activation = 'linear'))
    adam = keras.optimizers.Adam( lr = config.lr )


    model.compile(loss='mean_squared_error', optimizer=adam)
    

    return model
