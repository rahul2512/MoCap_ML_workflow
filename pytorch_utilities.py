from itertools import chain
import time
import pandas as pd, numpy as np, seaborn as sns, keras, random, xgboost as xgb, itertools
import matplotlib.pyplot as plt, sys, copy, scipy, joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
from scipy.interpolate import interp1d
from barchart_err import barchart_error, barchart_params
from tensorflow.keras import backend as K
from keras.regularizers import l2
# from keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Flatten, ConvLSTM1D, Input
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Flatten, ConvLSTM1D, Input, Masking
try:
    from keras.layers.convolutional import Conv1D, MaxPooling1D
except:
    from keras.layers import Conv1D, MaxPooling1D
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from tensorflow.keras import layers
from tensorflow.keras import backend as K
@keras.saving.register_keras_serializable()
def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))    
@keras.saving.register_keras_serializable()
def rmse(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))    

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


NN_name_map = {
    "LM": "Linear model",
    "FFNN": "Feedforward Neural Networks",
    "NN": "Feedforward Neural Networks",
    "VRNN": "Vanilla Recurrent Neural Networks",
    "LSTM": "Long Short-Term Memory",
    "GRU": "Gated Recurrent Unit",
    "CNN": "Convolutional Neural Networks",
    "RF": "Random Forests",
    "rf": "Random Forests",
    "XGBR": "Extreme Gradient Boosting",
    "xgbr": "Extreme Gradient Boosting",
    "CNNLSTM": "CNNLSTM",
    "convLSTM": "Convolutional LSTM",
}

### Initite NN model
def initiate_NN_model(ML_opt):
    print(ML_opt)
    model = keras.Sequential()
    if ML_opt['missing_marker']:
        model.add(Masking(mask_value=-99999, input_shape=(ML_opt['inp_dim'],)))
        model.add(Dense(ML_opt['num_nodes'], activation=ML_opt['act']))
        print('using masking layer....')
    else:
        model.add(Dense(ML_opt['num_nodes'], input_shape=(ML_opt['inp_dim'],), activation=ML_opt['act']))
    for i in range(ML_opt['H_layer']):
        model.add(Dense(ML_opt['num_nodes'], activation=ML_opt['act'],kernel_initializer=ML_opt['kinit']))
        # model.add(Flatten())
        model.add(Dropout(ML_opt['p']))
    model.add(Dense(ML_opt['out_dim'], activation=ML_opt['final_act']))
    opt = ML_opt['optim'](learning_rate = ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model

def initiate_LM(ML_opt):
    print(ML_opt)
    inputs = Input(shape=(ML_opt['inp_dim'],))
    outputs = Dense(ML_opt['out_dim'])(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    opt = ML_opt['optim'](learning_rate = ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model

def initiate_LR_model(ML_opt):  
    print(ML_opt)
    model = keras.Sequential()
    model.add(Dense(ML_opt['out_dim'], input_shape=(ML_opt['inp_dim'],), activation='sigmoid'))
    opt = ML_opt['optim'](learning_rate = ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model

## ENcoder-decoder architechutre of RNN
## consists of two recurrent neural networks (RNN) that act as an encoder and a decoder pair. 
## The encoder maps a variable-length source sequence to a fixed-length vector, and
## the decoder maps the vector representation back to a variable-length target sequence.
## https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
## https://www.kaggle.com/code/kankanashukla/types-of-lstm
## https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

def initiate_RNN_model(ML_opt): 
    print(ML_opt)
    # dropout_layer = Dropout(rate=p_drop)
    model = tf.keras.Sequential()
    model.add(Input(shape=(None, ML_opt['inp_dim'])))
    units = int(ML_opt['num_nodes']) ## for unknown reasons, it treats units as non-int
    #inputs: A 3D tensor with shape [batch, timesteps, feature]
    rnn_layers = {'SimpleRNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU, 'BSimpleRNN': SimpleRNN, 'BLSTM': LSTM, 'BGRU': GRU}
    base_layer = rnn_layers[ML_opt['NN_variant']]
    if ML_opt['H_layer'] == 0 :
        layer = base_layer(units=ML_opt['num_nodes'], activation=ML_opt['act'], dropout = ML_opt['p'], 
                           kernel_initializer=ML_opt['kinit'], return_sequences=False)
        if ML_opt['NN_variant'].startswith('B'): ## not using this appraoch anymore
            layer = Bidirectional(layer)
        model.add(layer)
        
    elif ML_opt['H_layer'] > 0 :     
        for _ in range(ML_opt['H_layer']):
            layer = base_layer(units=ML_opt['num_nodes'], activation=ML_opt['act'], dropout = ML_opt['p'], 
                               kernel_initializer=ML_opt['kinit'], return_sequences=True)
            if ML_opt['NN_variant'].startswith('B'):
                layer = Bidirectional(layer)
            model.add(layer)
        final_layer = base_layer(units=ML_opt['num_nodes']//2, activation=ML_opt['act'], dropout = ML_opt['p'], 
                                 kernel_initializer=ML_opt['kinit'], return_sequences=False)
        if ML_opt['NN_variant'].startswith('B'):
            final_layer = Bidirectional(final_layer)
        model.add(final_layer)

    model.add(Dense(ML_opt['out_dim']))  
    opt = ML_opt['optim'](learning_rate = ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model


def initiate_CNN_model(ML_opt): 
    print(ML_opt)
    model = keras.Sequential()
    #  (batch_size, timesteps, input_dim) -- input of conv1D layers
    model.add(Conv1D(filters = int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), strides=int(ML_opt['stride']), 
                     activation=ML_opt['act'], kernel_initializer=ML_opt['kinit'], padding = 'same',
                     input_shape=(ML_opt['t_dim'], ML_opt['inp_dim'])))
    model.add(MaxPooling1D(pool_size=int(ML_opt['pool_size']), padding = 'same'))
    # (batch_size, new_timesteps, filters), where new_timesteps is the length of the output sequence after applying the convolutional operation

    for i in range(ML_opt['H_layer']-1): 
        model.add(Conv1D(filters=2*int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), activation=ML_opt['act'], padding = 'same'))
        model.add(MaxPooling1D(pool_size=int(ML_opt['pool_size']), padding = 'same'))

    model.add(Flatten())
    ## dropout can be used below but we are not using it here
    model.add(Dense(50, activation=ML_opt['act']))
    model.add(Dense(ML_opt['out_dim']))
    opt = ML_opt['optim'](learning_rate = ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model




def resnet_block(input_layer, filters, kernel_size=3, stride=1):
    # First layer of the block
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_layer])
    x = Activation('relu')(x)
    return x

def initiate_ResNet_model(ML_opt):
    ML_opt['input_height'] = 2
    ML_opt['input_width'] = 2    
    ML_opt['input_channels'] = 2
    input_layer = Input(shape=(ML_opt['input_height'], ML_opt['input_width'], ML_opt['input_channels']))
    # Initial Convolution and Pooling
    x = Conv2D(filters=int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), strides=int(ML_opt['stride']), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(ML_opt['act'])(x)
    x = MaxPooling2D(pool_size=int(ML_opt['pool_size']), padding='same')(x)
    # Adding ResNet Blocks
    for i in range(ML_opt['H_layer'] - 1):
        x = resnet_block(x, filters=2*int(ML_opt['num_nodes']))
    # Output Layer
    x = Flatten()(x)
    x = Dense(50, activation=ML_opt['act'])(x)
    output_layer = Dense(ML_opt['out_dim'], activation='softmax')(x)
    # Compile Model
    model = Model(inputs=input_layer, outputs=output_layer)
    opt = ML_opt['optim'](learning_rate=ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model

def initiate_CNNLSTM_model(ML_opt): 
    print(ML_opt)
    inp_dim = int(ML_opt['inp_dim'])
    out_dim = int(ML_opt['out_dim'])
    t_dim   = int(ML_opt['t_dim'])
    nbr_Hlayer = int(ML_opt['H_layer'])
    batch_size = int(ML_opt['batch_size'])
    filt_size = int(ML_opt['filt_size'])
    stride = int(ML_opt['stride'])
    LSTM_units = int(ML_opt['LSTM_units'])
    pool_size = int(ML_opt['pool_size'])
    units =  int(ML_opt['num_nodes'])

    model = keras.Sequential()
    model.add(Input(shape=(t_dim, inp_dim)))  # Explicitly define the input shape
    model.add(Conv1D(filters=units, kernel_size=filt_size, strides=stride, activation=ML_opt['act'], kernel_initializer=ML_opt['kinit'], padding='same'))    
    model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
    model.add(LSTM(units=LSTM_units, return_sequences=nbr_Hlayer > 1))

    if nbr_Hlayer > 1:
        for i in range(nbr_Hlayer - 1): 
            model.add(Conv1D(filters=units, kernel_size=filt_size, strides=stride, activation=ML_opt['act'], kernel_initializer=ML_opt['kinit'], padding='same'))
            model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
            model.add(LSTM(units=LSTM_units, return_sequences=(i < nbr_Hlayer - 1)))  # True for all but the last LSTM layer

    model.add(Dense(out_dim))
    opt = ML_opt['optim'](learning_rate = ML_opt['lr'])
    model.compile(loss=ML_opt['loss'], optimizer=opt, metrics=ML_opt['metric'])
    return model



def initiate_ConvLSTM_model(ML_opt):
    #https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
    print(ML_opt)
    ML_opt['H_layer'] = 1
    # ML_opt['stride'] = 1
    model = keras.Sequential()
    if ML_opt['H_layer'] == 1 :
        model.add(ConvLSTM1D(filters = int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), strides=int(ML_opt['stride']),
                             activation=ML_opt['act'], kernel_initializer=ML_opt['kinit'], padding='same', return_sequences=False,
                             input_shape=(ML_opt['t_dim'], ML_opt['inp_dim'], 1)))
        model.add(MaxPooling1D(pool_size=int(ML_opt['pool_size']), padding = 'same'))
    elif ML_opt['H_layer'] > 1 :
        model.add(ConvLSTM1D(filters = int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), strides=int(ML_opt['stride']),
                             activation=ML_opt['act'], kernel_initializer=ML_opt['kinit'], padding='same', return_sequences=True,
                             input_shape=(ML_opt['t_dim'], ML_opt['inp_dim'], 1)))
        for i in range(ML_opt['H_layer']-2):
            model.add(ConvLSTM1D(filters=int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), activation=ML_opt['act'], 
                                 padding = 'same', return_sequences=True))
        model.add(ConvLSTM1D(filters=int(ML_opt['num_nodes']), kernel_size=int(ML_opt['filt_size']), activation=ML_opt['act'], 
                             padding = 'same', return_sequences=False))
    else:
        sys.exit()

    model.add(Flatten())
    # model.add(Dense(50, activation='relu',kernel_initializer="glorot_normal"))
    model.add(Dense(ML_opt['out_dim']))
    model.compile(loss=ML_opt['loss'], optimizer=ML_opt['optim'](learning_rate=ML_opt['lr']), metrics=ML_opt['metric'])
    print("Initialised ConvLSTM .....")
    print(model.summary())
    return model


######################################
## Randomforest
######################################
def rf(ML_opt,X_Train, Y_Train, X_val, Y_val):
    model = RandomForestRegressor(n_estimators=ML_opt['n_estimators'], verbose = ML_opt['verbose'], max_features=ML_opt['max_features'], 
                                  max_depth=ML_opt['max_depth'], min_samples_split=ML_opt['min_samples_split'], min_samples_leaf=ML_opt['min_samples_leaf'], 
                                  bootstrap=ML_opt['bootstrap'], criterion=ML_opt['criterion'], n_jobs=4)
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

######################################
## xgboost
######################################
def xgbr(ML_opt, X_Train, Y_Train, X_val, Y_val):
    model = xgb.XGBRegressor(objective=ML_opt['objective'], 
                             n_estimators=ML_opt['n_estimators'], 
                             learning_rate=ML_opt['learning_rate'], 
                             max_depth=ML_opt['max_depth'], 
                             reg_alpha=ML_opt['alpha'], 
                             reg_lambda=ML_opt['lambda1'])
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

######################################
#GradientBoostingRegressor
###################################
def GBRT(ML_opt, X_Train, Y_Train, X_val, Y_val):
    model = GradientBoostingRegressor(n_estimators=ML_opt['n_estimators'], verbose = ML_opt['verbose'], 
                                      max_features=ML_opt['max_features'], max_depth=ML_opt['max_depth'], 
                                      min_samples_split=ML_opt['min_samples_split'], min_samples_leaf=ML_opt['min_samples_leaf'], 
                                      loss=ML_opt['loss'])
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

######################################
#SVR
###################################
from sklearn.svm import SVR
def SVR_model(X_Train, Y_Train, X_val, Y_val):
    model = SVR(kernel='rbf') 
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, y_pred)
    print("Validation MSE --" , mse)
    return model

###################
## Transformer code
###################
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def transformer(input_shape, out_dim, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, mlp_dropout, dropout):
    mlp_units = np.array(eval(mlp_units))
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(out_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="mae", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=["mse"],)
    return model


#### Code below generates a .txt file with row as the list of hypermeters 
#### for a given NN and then that NN hypermeters were cross-validated on cluster
def write(hyperparameters, file_name):
    with open(f'./hyperparameters/hyperparam_{file_name}.txt', 'w') as f:
        # print(*hyperparameters.keys(), file=f)
        print(','.join(hyperparameters.keys()), file=f)
        for values in itertools.product(*hyperparameters.values()):
            print(','.join(map(str, values)), file=f)
            # print(*values, file=f)
    return None

def hyper_param_rf():
    hyperparameters = {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True],
        'criterion': ['squared_error', 'absolute_error'],
        'norm_out': [0]
    }
    write(hyperparameters, 'rf')



def hyper_param_xgbr():
    hyperparameters = {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.005],
        'max_depth': [10, 20, 30, 40, 50],
        'objective': ['reg:squarederror'],
        'alpha': [0.1, 0.2],
        'lambda1': [0.1, 0.2],
        'norm_out': [0]
    }
    write(hyperparameters, 'xgbr')

def hyper_param_GBRT():
    hyperparameters = {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 30, 50, 70, 90, 110],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'loss': ['squared_error', 'absolute_error'],
        'norm_out': [0]
    }
    write(hyperparameters, 'GBRT')

def hyper_param_transformer():
    hyperparameters = {
        'head_size': [64],
        'num_heads': [8],
        'ff_dim': [4],
        'num_transformer_blocks': [4, 6, 8],
        'mlp_units': ['[32]', '[32,32]', '[32,32,32]'],
        'mlp_dropout': [0.2],
        'epoch': [100, 200, 300, 500],
        'batch_size': [64, 128],
        'norm_out': [0, 1]
    }
    write(hyperparameters, 'transformer')

def hyper_param_NN():
    hyperparameters = {
        'optim': ['Adam', 'SGD'],
        'kinit': ['glorot_normal'],
        'batch_size': [64, 128],
        'epoch': [50, 100, 200],
        'act': ['relu', 'tanh', 'sigmoid'],
        'num_nodes': list(range(200, 1100, 200)),
        'H_layer': [2, 4, 6, 8],
        'metric': ['rmse'],
        'loss': ['mse', 'rmse'],
        'lr': [0.001, 0.005],
        'p': [0.1, 0.2],
        'regularizer_val': [0],
        'NN_variant': ['NN'],
        'norm_out': [0, 1]
    }
    write(hyperparameters, 'NN')

def hyper_param_CNN():
    hyperparameters = {
        'optim': ['Adam', 'SGD', 'RMSprop'],
        'kinit': ['glorot_normal'],
        'batch_size': [64, 128],
        'epoch': [50, 100, 200],
        'act': ['relu', 'tanh', 'sigmoid'],
        'num_nodes': [32, 64],
        'H_layer': [1, 2],
        'metric': ['rmse'],
        'loss': ['mse', 'rmse'],
        'lr': [0.001, 0.005],
        'pool_size': [2],
        'regularizer_val': [0],
        'NN_variant': ['CNN'],
        'filt_size': [3],
        'stride': [1, 3],
        'norm_out': [0, 1]
    }
    write(hyperparameters, 'CNN')

def hyper_param_CNNLSTM():
    hyperparameters = {
        'optim': ['Adam', 'SGD', 'RMSprop'],
        'kinit': ['glorot_normal'],
        'batch_size': [64, 128],
        'epoch': [50, 100, 200],
        'act': ['relu', 'tanh', 'sigmoid'],
        'num_nodes': [32, 64],
        'H_layer': [1, 2],
        'metric': ['rmse'],
        'loss': ['mse', 'rmse'],
        'lr': [0.001, 0.005],
        'pool_size': [2],
        'regularizer_val': [0],
        'NN_variant': ['CNNLSTM'],
        'filt_size': [3],
        'stride': [1, 3],
        'LSTM_units': [128, 256],
        'norm_out': [0, 1]
    }
    write(hyperparameters, 'CNNLSTM')

def hyper_param_LM():
    hyperparameters = {
        'optim': ['Adam', 'SGD'],
        'kinit': ['glorot_normal', 'random_normal', 'he_normal'],
        'batch_size': [64, 256],
        'epoch': [50, 100, 200],
        'act': ['linear'],
        'num_nodes': [200],
        'H_layer': [0],
        'metric': ['rmse'],
        'loss': ['mse', 'rmse'],
        'lr': [0.001],
        'p': [0],
        'regularizer_val': [0],
        'NN_variant': ['LM'],
        'norm_out': [0, 1]
    }
    write(hyperparameters, 'LM')

def hyper_param_RNN():
    hyperparameters = {
        'NN_variant': ['SimpleRNN', 'LSTM', 'GRU'],
        'optim': ['Adam', 'RMSprop'],
        'kinit': ['glorot_normal'],
        'batch_size': [64, 128],
        'epoch': [50, 100, 200],
        'act': ['relu', 'tanh', 'sigmoid'],
        'num_nodes': [128, 256, 512],
        'H_layer': [0, 1, 2, 3],
        'metric': ['rmse'],
        'loss': ['mse', 'rmse'],
        'lr': [0.001],
        'p': [0.1, 0.2],
        'regularizer_val': [0],
        'norm_out': [0, 1]
    }
    write(hyperparameters, 'RNN')

# hyper_param_transformer()
# hyper_param_xgbr()
# hyper_param_rf()
# hyper_param_GBRT()
# hyper_param_LM()
# hyper_param_NN()
# hyper_param_RNN()
# hyper_param_CNN()
# hyper_param_CNNLSTM()