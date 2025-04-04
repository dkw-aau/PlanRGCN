import os

import pandas as pd
import matplotlib.pyplot as plt
from svm import normalize_target, coeff_determination,rmse, load_dataset_preserving_split
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1524)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print("dasdasdsa",e)

import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

#sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import NuSVR
#Keras imports
from keras import backend as K

from keras.layers import Input, Dense
from keras.optimizers import RMSprop, SGD, Adagrad,Adam,Adadelta,Adamax,Nadam
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Lambda, BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam

from time import time
import math

"""def get_metrics(model, scaler, x_data, y_true_data, label_set="Data"):
    y_pred = np.exp(scaler.inverse_transform(model.predict(x_data).reshape(-1, 1)))
    y_true_data = np.nan_to_num(y_true_data)
    y_pred = np.nan_to_num(y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true_data, y_pred))
    #rmse_val = rmse(y_true_data,y_pred)
    r2 = r2_score(y_true_data, y_pred)
    print("RMSE "+label_set, rmse_val)
    print("R2 SCORE "+label_set, r2)
    return rmse_val, r2"""
#if metrics are -1, it means that the predictions are too big.
def get_metrics(model, scaler, x_data, y_true_data, label_set="Data"):
    y_pred = np.exp(scaler.inverse_transform(model.predict(x_data).reshape(-1, 1)))
    try:
        rmse = np.sqrt(mean_squared_error(y_true_data, y_pred))
        r2 = r2_score(y_true_data, y_pred)
    except ValueError:
        rmse = np.finfo(np.float32).max
        r2 =np.finfo(np.float32).max
    print("RMSE "+label_set, rmse)
    print("R2 SCORE "+label_set, r2)
    return rmse, r2
def plot_history(history, metrics_list, start_at_epoch=0):
    plt.clf()
    for metric in metrics_list:
        plt.plot(history.history[metric][start_at_epoch:],label=metric)
    plt.legend()
    plt.title("Metrics by epochs(Start from epoch:{})".format(start_at_epoch))
    plt.savefig("plot"+str(time())+".png", format="png")
    plt.show()

def train_autoencoder(x_train, x_val, encoding_dim, verbose,batch_size=120, save_path=None):
    """
    Train an autoencoder for preprocessing data.
    """
    # Set callback functions to early stop training and save the best model so far
    callbacks_best = [EarlyStopping(monitor='val_loss', patience=10),
                      ]
    # this is the size of our encoded representations
    # this is our input placeholder
    input_img = Input(shape=(x_train.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(x_train.shape[1], activation='linear')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer=Nadam(learning_rate=0.0001), loss='mse')
    print("Autoencoder Summary")
    print(autoencoder.summary())
    autoencoder.fit(x_train, x_train,
                    epochs=300,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_val, x_val),
                    verbose=verbose,
                    callbacks=callbacks_best)

    plot_history(autoencoder.history, metrics_list=['loss','val_loss'], start_at_epoch=10)
    #autoencoder.save("/code/data/11Dec/NN/autoencoder_newdata.hdf5")
    if save_path != None:
        autoencoder.save(save_path)
    elif 'datapath' in os.environ.keys():
        autoencoder.save(os.environ['datapath']+'/'+os.environ['experiment_name']+ "/NN/autoencoder_newdata.hdf5")

    return autoencoder

# Set callback functions to early stop training and save the best model so far
def build_train_ann_with_aec(autoencoder, x_train, y_train, x_val, y_val, n1, n2, n3, epochs, optimizer, dropout,verbose=False, save_path=None):
    if save_path != None:
        callbacks_best = [EarlyStopping(monitor='val_loss', patience=20),
                      #ModelCheckpoint(filepath='/code/data/11Dec/NN/bestm_newdata.h5'.format(n1,n2,n3),
                      ModelCheckpoint(filepath= save_path,
                                      monitor='val_loss', save_best_only=True
                                      )]
    elif 'datapath' in os.environ.keys():
        callbacks_best = [EarlyStopping(monitor='val_loss', patience=20),
                      #ModelCheckpoint(filepath='/code/data/11Dec/NN/bestm_newdata.h5'.format(n1,n2,n3),
                      ModelCheckpoint(filepath=os.environ['datapath']+'/'+os.environ['experiment_name']+'/NN/bestm_newdata.h5'.format(n1,n2,n3),
                                      monitor='val_loss', save_best_only=True
                                      )]
    else:
        callbacks_best = [EarlyStopping(monitor='val_loss', patience=20)
                      ]

    model = Sequential()
    model.add(autoencoder.layers[0])
    model.add(autoencoder.layers[1])
    model.add(Dense(n1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', coeff_determination, rmse])
    t0=time()
    print("before train: Init time: {}".format(round(t0,3)))
    history = model.fit(x_train.values, y_train,
                        epochs=epochs,
                        batch_size=120,
                        validation_data=(x_val.values, y_val),
                        callbacks=callbacks_best,
                        verbose=verbose
                        )
    t1=time()
    print("after train, finish time: {}".format(round(t1,3)))
    print("training time {}",format(round(t1-t0, 3)))
    print(model.summary())

    return model, history, round(t1-t0, 3)

# Set callback functions to early stop training and save the best model so far
def build_train_ann(x_train, y_train, x_val, y_val, n1, n2, n3, epochs, optimizer, dropout, verbose=False):

    callbacks_best = [EarlyStopping(monitor='val_loss', patience=20),
                      #ModelCheckpoint(filepath='/code/data/11Dec/NN/bestm_newdata.h5'.format(n1,n2,n3),
                      ModelCheckpoint(filepath=os.environ['datapath']+'/'+os.environ['experiment_name']+'/NN/bestm_newdata.h5'.format(n1,n2,n3),
                                      monitor='val_loss', save_best_only=True
                                      )]

    model = Sequential()
    model.add(Dense(n1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', coeff_determination, rmse])
    t0=time()
    print("before train: Init time: {}".format(round(t0,3)))
    history = model.fit(x_train.values, y_train,
                        epochs=epochs,
                        batch_size=120,
                        validation_data=(x_val.values, y_val),
                        callbacks=callbacks_best,
                        verbose=verbose
                        )
    t1=time()
    print("after train, finish time: {}".format(round(t1,3)))
    print("training time {}",format(round(t1-t0, 3)))
    print(model.summary())

    return model, history, round(t1-t0, 3)




def train_complete_model2(df_train, data_val, data_test, with_aec=True, verbose_train=False,drop_columns=[], units=[85, 60, 60], optimizer=None, train_aec=True,aec_units=20):

    #print columns
    print("Columns datasets", df_train.columns)
    #transform data and get in log scale
    x_train, x_val, x_test , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log = scale_log_data_targets(df_train, data_val, data_test)
    print("Shape datasets x: {}".format(x_train.shape))
    print("Shape datasets xval: {}".format(x_val.shape))
    print("Shape datasets xtest: {}".format(x_test.shape))
    print("Columns datasets after normalize.", x_train.columns)
    # scale target using StandarSacaler
    scalery, y_train_log_std, y_val_log_std, y_test_log_std = normalize_target(y_train_log, y_val_log, y_test_log)

    print(
        "Dimensiones de los datos: Cant ejemplos en train:{}, Val: {} Test: {}".
        format(x_train.shape, x_val.shape, x_test.shape)
    )
    #Train autoencoder:
    if train_aec:
        autoencoder = train_autoencoder(x_train, x_val, aec_units, verbose=False)
    #else:
    #    autoencoder = load_model("models_newdata/autoencoder_newdata.hdf5")
    #Train model ANN
    epochs=450
    dropout=0.25
    if optimizer is None:
        optimizer=Adam(learning_rate=0.00015)
    if with_aec:

        model, history, time_training = build_train_ann_with_aec(
            autoencoder,
            x_train,
            y_train_log_std,
            x_val,
            y_val_log_std,
            units[0],
            units[1],
            units[2],
            epochs,
            optimizer,
            dropout,
            verbose=verbose_train)
    else:
        model, history, time_training = build_train_ann(
            x_train,
            y_train_log_std,
            x_val,
            y_val_log_std,
            units[0],
            units[1],
            units[2],
            epochs,
            optimizer,
            dropout,
            verbose=verbose_train)
    #load weights for best model.
    #     best_model = build_best_model(units=[85, 60, 60])
    #     best_model.load_weights('models_legacy/bestm_legacydata.hdf5')
    rmse_train, r2_train = get_metrics(model, scalery, x_train, y_train, label_set=" Train ")
    rmse_val, r2_val     = get_metrics(model, scalery, x_val, y_val, label_set=" Val ")
    rmse_test, r2_test   = get_metrics(model, scalery, x_test, y_test, label_set=" Test ")
    plot_history(history, metrics_list=['loss','val_loss'],start_at_epoch=5)
    return model, scalery, rmse_train, rmse_val, rmse_test, r2_train, r2_val, r2_test, time_training

def executar(x_train,x_val,x_test,train_aec=True, with_aec=True):
    values_rmse = []
    values_r2 = []
    best_rmse = None
    #import random
    stats = pd.DataFrame(
        {
            "rmse_train":[],
            "rmse_val"  :[],
            "rmse_test" :[],
            "r2_train"  :[],
            "r2_val"    :[],
            "r2_test"   :[],
        }
    )
    stats = {
            "rmse_train":[],
            "rmse_val"  :[],
            "rmse_test" :[],
            "r2_train"  :[],
            "r2_val"    :[],
            "r2_test"   :[],
    }
    for i in range(0,10):#10
        units = [340, 380, 340]
        print(units)
        optimizer=Adam(learning_rate=0.00015)
        model, scalery, rmse_train, rmse_val, rmse_test, r2_train, r2_val, r2_test, time_training = train_complete_model2(x_train, x_val, x_test, verbose_train=False, optimizer=optimizer, drop_columns=[],units=units,train_aec=train_aec, with_aec=with_aec, aec_units=30)
        """stats = stats.append({
            "rmse_train":rmse_train,
            "rmse_val"  :rmse_val,
            "rmse_test" :rmse_test,
            "r2_train"  :r2_train,
            "r2_val"    :r2_val,
            "r2_test"   :r2_test,
        },
            ignore_index=True)"""
        stats['rmse_train'].append(rmse_train)
        stats['rmse_val'].append(rmse_val)
        stats['rmse_test'].append(rmse_test)
        stats['r2_train'].append(r2_train)
        stats['r2_val'].append(r2_val)
        stats['r2_test'].append(r2_test)
        values_rmse.append(rmse_test)
        values_r2.append(r2_test)
        if (best_rmse == None or rmse_test < best_rmse):
            best_model = model
            best_scalery = scalery
            best_rmse = rmse_test
    stats = pd.DataFrame.from_dict(stats)
    print("RMSE mean 10 rams: {}".format(np.mean(values_rmse)))
    print("Best RMSE: {}".format(best_rmse))
    print("R2 mean 10 rams: {}".format(np.mean(values_r2)))
    return stats, best_model, best_scalery

from util import convert_string2dict, get_sum_values,selectivity,get_hist_value,pred_2_hist,get_filter_by_type
from util import filter_only_string_non_empty,uri_2_index_seq,get_joins, normalizaAlgebra, joinAlgebraGPM, printSTDVARMEAN, scale_log_data_targets

def process_extra(data_tpf):
    data_tpf['predicates'] = data_tpf['predicates'].apply(lambda x: filter_only_string_non_empty(x))
    data_tpf['joinsv1'] =  data_tpf['joins'].apply(lambda x: get_joins(x))
    data_tpf['joins_count'] = data_tpf['joinsv1'].apply(lambda x: len(x))
    data_tpf['predicates_select'] = data_tpf['predicates'].apply(lambda x: pred_2_hist(x))

    data_tpf['filter_uri'] = data_tpf['predicates_select'].apply(lambda x: get_filter_by_type(x, 'uri'))
    data_tpf['filter_num'] = data_tpf['predicates_select'].apply(lambda x: get_filter_by_type(x, 'num'))
    data_tpf['filter_literal'] = data_tpf['predicates_select'].apply(lambda x: get_filter_by_type(x, 'literal'))
    data_tpf_clean = data_tpf[['id','joins_count','filter_uri','filter_num','filter_literal']]
    return data_tpf_clean

def save_prediction(xs,scaler,model,path_to_save):
    
    x_pred = xs.drop(columns=['time'])
    #TODO: in svm np.exp is not used.
    y_pred_train = np.exp(scaler.inverse_transform(model.predict( x_pred )))
    xs['nn_prediction'] = y_pred_train
    remove_columns = [x for x in xs.columns if not x in ['time','nn_prediction']]
    x_pred = xs.drop(columns=remove_columns)
    x_pred.to_csv(path_to_save)

if __name__ == "__main__":
    #PATH = '/code/data/11Dec/data/'
    #query_log_path = '/code/data/datasetlsq_30000.csv'
    PATH = os.environ['datapath'] +'/'+ os.environ['experiment_name']+'/data'
    query_log_path = os.environ['datapath']+'/datasetlsq_30000.csv'

    train_df, val_df, test_df = load_dataset_preserving_split(PATH, query_log_path)

    data_tpf = pd.read_csv(PATH + '/extra.csv', delimiter="ᶶ")
    data_tpf = data_tpf.drop(columns=["Unnamed: 4",'execTime'])
    data_tpf_clean = process_extra(data_tpf)

    X_train_extended = train_df.merge(data_tpf_clean, left_on='id', right_on='id')
    X_val_extended = val_df.merge(data_tpf_clean, left_on='id', right_on='id')
    X_test_extended = test_df.merge(data_tpf_clean, left_on='id', right_on='id')

    X_train_extended = X_train_extended.set_index('id')
    X_val_extended = X_val_extended.set_index('id')
    X_test_extended = X_test_extended.set_index('id')
    ###
    X_train_extended = X_train_extended.drop(columns=['join'])
    X_val_extended = X_val_extended.drop(columns=['join'])
    X_test_extended = X_test_extended.drop(columns=['join'])

    scaled_df_train, scaled_df_val, scaled_df_test = normalizaAlgebra(X_train_extended, X_val_extended, X_test_extended)
    col_gpm = ['time','cls_0', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_6', 'cls_7', 'cls_8', 'cls_9', 'cls_10', 'cls_11', 'cls_12',
     'cls_13', 'cls_14', 'cls_15', 'cls_16', 'cls_17', 'cls_18', 'cls_19', 'cls_20', 'cls_21', 'cls_22', 'cls_23', 'cls_24']
    x_train, x_val, x_test = joinAlgebraGPM(scaled_df_train, scaled_df_val, scaled_df_test, X_train_extended[col_gpm], X_val_extended[col_gpm], X_test_extended[col_gpm])
    #print(x_train.head(5))

    #should be outcommented for actual run
    stats, model, scaler = executar(x_train,x_val,x_test,train_aec=True)
    print("AEC + ANN")
    print(stats)
    printSTDVARMEAN(stats, "AEC + ANN")

    #stats, model, scaler = executar(train_df,val_df,test_df,train_aec=False,with_aec=False)
    save_prediction(x_train,scaler,model,os.environ['datapath'] +'/'+ os.environ['experiment_name']+"/results/nn_train_pred.csv")
    save_prediction(x_val,scaler,model,os.environ['datapath'] +'/'+ os.environ['experiment_name']+"/results/nn_val_pred.csv")
    save_prediction(x_test,scaler,model,os.environ['datapath'] +'/'+ os.environ['experiment_name']+"/results/nn_test_pred.csv")
    #model.predict(train_df.iloc[0])