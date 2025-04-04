import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")
import tensorflow as tf
from keras.optimizers import RMSprop, SGD, Adagrad, Adam, Adadelta, Adamax, Nadam
from keras.models import Sequential, Model
from keras.layers import Dropout
import os
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from time import time
from sklearn.preprocessing import StandardScaler
#from keras import backend as K
import tensorflow.keras.backend as K
import pickle
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1524)],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print("dasdasdsa", e)


class NNTrainer:
    def __init__(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, resultDir: str
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.resultDir = resultDir

    def trainer(self):
        stats, model, scaler = self.executar(
            self.train, self.val, self.test, train_aec=True
        )
        model.save(os.path.join(self.resultDir,'nn.keras'))
        with open(os.path.join(self.resultDir,'scaler.pickle'), 'wb') as f:
            pickle.dump(scaler, f)
        self.model = model
        self.scaler = scaler
        print("AEC + ANN")
        print(stats)
        NNutils.printSTDVARMEAN(stats, "AEC + ANN")
        self.save_prediction(
            self.train, scaler, model, self.resultDir + "nn_train_pred.csv"
        )
        self.save_prediction(
            self.val, scaler, model, self.resultDir + "nn_val_pred.csv"
        )
        self.save_prediction(
            self.test, scaler, model, self.resultDir + "nn_test_pred.csv"
        )
        
    def predict_trained(self, test, path):
        
        model = self.model
        scaler = self.scaler
        
        self.save_prediction(
            test, scaler, model, path
        )

    def save_prediction(self, xs, scaler, model, path_to_save):
        x_pred = xs.drop(columns=["time"])
        # TODO: in svm np.exp is not used.
        start = time()
        y_pred_train = np.exp(scaler.inverse_transform(model.predict(x_pred)))
        dur = time()-start
        dur = dur/len(xs)
        xs["nn_prediction"] = y_pred_train
        remove_columns = [x for x in xs.columns if not x in ["time", "nn_prediction"]]
        x_pred = xs.drop(columns=remove_columns)
        x_pred['inference'] = dur
        x_pred.to_csv(path_to_save)

    def executar(self, x_train, x_val, x_test, train_aec=True, with_aec=True):
        values_rmse = []
        values_r2 = []
        best_rmse = None
        # import random
        stats = pd.DataFrame(
            {
                "rmse_train": [],
                "rmse_val": [],
                "rmse_test": [],
                "r2_train": [],
                "r2_val": [],
                "r2_test": [],
            }
        )
        stats = {
            "rmse_train": [],
            "rmse_val": [],
            "rmse_test": [],
            "r2_train": [],
            "r2_val": [],
            "r2_test": [],
        }
        for i in range(0, 10):  # 10
            units = [340, 380, 340]
            print(units)
            optimizer = Adam(learning_rate=0.00015)
            (
                model,
                scalery,
                rmse_train,
                rmse_val,
                rmse_test,
                r2_train,
                r2_val,
                r2_test,
                time_training,
            ) = self.train_complete_model2(
                x_train,
                x_val,
                x_test,
                verbose_train=False,
                optimizer=optimizer,
                drop_columns=[],
                units=units,
                train_aec=train_aec,
                with_aec=with_aec,
                aec_units=30,
            )
            stats["rmse_train"].append(rmse_train)
            stats["rmse_val"].append(rmse_val)
            stats["rmse_test"].append(rmse_test)
            stats["r2_train"].append(r2_train)
            stats["r2_val"].append(r2_val)
            stats["r2_test"].append(r2_test)
            values_rmse.append(rmse_test)
            values_r2.append(r2_test)
            if best_rmse == None or rmse_test < best_rmse:
                best_model = model
                best_scalery = scalery
                best_rmse = rmse_test
        stats = pd.DataFrame.from_dict(stats)
        print("RMSE mean 10 rams: {}".format(np.mean(values_rmse)))
        print("Best RMSE: {}".format(best_rmse))
        print("R2 mean 10 rams: {}".format(np.mean(values_r2)))
        return stats, best_model, best_scalery

    def train_complete_model2(
        self,
        df_train,
        data_val,
        data_test,
        with_aec=True,
        verbose_train=False,
        drop_columns=[],
        units=[85, 60, 60],
        optimizer=None,
        train_aec=True,
        aec_units=20,
    ):
        # print columns
        print("Columns datasets", df_train.columns)
        # transform data and get in log scale
        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            y_train_log,
            y_val_log,
            y_test_log,
        ) = NNutils.scale_log_data_targets(df_train, data_val, data_test)
        print("Shape datasets x: {}".format(x_train.shape))
        print("Shape datasets xval: {}".format(x_val.shape))
        print("Shape datasets xtest: {}".format(x_test.shape))
        print("Columns datasets after normalize.", x_train.columns)
        # scale target using StandarSacaler
        (
            scalery,
            y_train_log_std,
            y_val_log_std,
            y_test_log_std,
        ) = NNutils.normalize_target(y_train_log, y_val_log, y_test_log)

        print(
            "Dimensiones de los datos: Cant ejemplos en train:{}, Val: {} Test: {}".format(
                x_train.shape, x_val.shape, x_test.shape
            )
        )
        # Train autoencoder:
        if train_aec:
            autoencoder = self.train_autoencoder(
                x_train, x_val, aec_units, verbose=False
            )
        # else:
        #    autoencoder = load_model("models_newdata/autoencoder_newdata.hdf5")
        # Train model ANN
        epochs = 450
        dropout = 0.25
        if optimizer is None:
            optimizer = Adam(learning_rate=0.00015)
        if with_aec:
            model, history, time_training = self.build_train_ann_with_aec(
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
                verbose=verbose_train,
            )
        else:
            model, history, time_training = self.build_train_ann(
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
                verbose=verbose_train,
            )
        # load weights for best model.
        #     best_model = build_best_model(units=[85, 60, 60])
        #     best_model.load_weights('models_legacy/bestm_legacydata.hdf5')
        rmse_train, r2_train = NNutils.get_metrics(
            model, scalery, x_train, y_train, label_set=" Train "
        )
        rmse_val, r2_val = NNutils.get_metrics(
            model, scalery, x_val, y_val, label_set=" Val "
        )
        rmse_test, r2_test = NNutils.get_metrics(
            model, scalery, x_test, y_test, label_set=" Test "
        )
        NNutils.plot_history(
            history, metrics_list=["loss", "val_loss"], start_at_epoch=0
        )
        
        return (
            model,
            scalery,
            rmse_train,
            rmse_val,
            rmse_test,
            r2_train,
            r2_val,
            r2_test,
            time_training,
        )

    # Set callback functions to early stop training and save the best model so far
    def build_train_ann(
        x_train,
        y_train,
        x_val,
        y_val,
        n1,
        n2,
        n3,
        epochs,
        optimizer,
        dropout,
        verbose=False,
    ):
        callbacks_best = [
            EarlyStopping(monitor="val_loss", patience=20),
            # ModelCheckpoint(filepath='/code/data/11Dec/NN/bestm_newdata.h5'.format(n1,n2,n3),
            ModelCheckpoint(
                filepath=os.environ["datapath"]
                + "/"
                + os.environ["experiment_name"]
                + "/NN/bestm_newdata.h5".format(n1, n2, n3),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        model = Sequential()
        model.add(Dense(n1, activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(n2, activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(n3, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=["mae", NNutils.coeff_determination, NNutils.rmse],
        )
        t0 = time()
        print("before train: Init time: {}".format(round(t0, 3)))
        history = model.fit(
            x_train.values,
            y_train,
            epochs=epochs,
            batch_size=120,
            validation_data=(x_val.values, y_val),
            callbacks=callbacks_best,
            verbose=verbose,
        )
        t1 = time()
        print("after train, finish time: {}".format(round(t1, 3)))
        print("training time {}", format(round(t1 - t0, 3)))
        print(model.summary())

        return model, history, round(t1 - t0, 3)

    # Set callback functions to early stop training and save the best model so far
    def build_train_ann_with_aec(
        self,
        autoencoder,
        x_train,
        y_train,
        x_val,
        y_val,
        n1,
        n2,
        n3,
        epochs,
        optimizer,
        dropout,
        verbose=False,
        save_path=None,
    ):
        if save_path != None:
            callbacks_best = [
                EarlyStopping(monitor="val_loss", patience=20),
                # ModelCheckpoint(filepath='/code/data/11Dec/NN/bestm_newdata.h5'.format(n1,n2,n3),
                ModelCheckpoint(
                    filepath=save_path, monitor="val_loss", save_best_only=True
                ),
            ]
        elif "datapath" in os.environ.keys():
            callbacks_best = [
                EarlyStopping(monitor="val_loss", patience=20),
                # ModelCheckpoint(filepath='/code/data/11Dec/NN/bestm_newdata.h5'.format(n1,n2,n3),
                ModelCheckpoint(
                    filepath=os.environ["datapath"]
                    + "/"
                    + os.environ["experiment_name"]
                    + "/NN/bestm_newdata.h5".format(n1, n2, n3),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]
        else:
            callbacks_best = [EarlyStopping(monitor="val_loss", patience=20)]

        model = Sequential()
        model.add(autoencoder.layers[0])
        model.add(autoencoder.layers[1])
        model.add(Dense(n1, activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(n2, activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(n3, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=["mae", NNutils.coeff_determination, NNutils.rmse],
        )
        t0 = time()
        print("before train: Init time: {}".format(round(t0, 3)))
        history = model.fit(
            x_train.values,
            y_train,
            epochs=epochs,
            batch_size=120,
            validation_data=(x_val.values, y_val),
            callbacks=callbacks_best,
            verbose=verbose,
        )
        t1 = time()
        print("after train, finish time: {}".format(round(t1, 3)))
        print("training time {}", format(round(t1 - t0, 3)))
        print(model.summary())

        return model, history, round(t1 - t0, 3)

    def train_autoencoder(
        self, x_train, x_val, encoding_dim, verbose, batch_size=120, save_path=None
    ):
        """
        Train an autoencoder for preprocessing data.
        """
        # Set callback functions to early stop training and save the best model so far
        callbacks_best = [
            EarlyStopping(monitor="val_loss", patience=10),
        ]
        # this is the size of our encoded representations
        # this is our input placeholder
        input_img = Input(shape=(x_train.shape[1],))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation="relu")(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(x_train.shape[1], activation="linear")(encoded)

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

        autoencoder.compile(optimizer=Nadam(learning_rate=0.0001), loss="mse")
        print("Autoencoder Summary")
        print(autoencoder.summary())
        autoencoder.fit(
            x_train,
            x_train,
            epochs=300,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val, x_val),
            verbose=verbose,
            callbacks=callbacks_best,
        )

        NNutils.plot_history(
            autoencoder.history, metrics_list=["loss", "val_loss"], start_at_epoch=10
        )
        # autoencoder.save("/code/data/11Dec/NN/autoencoder_newdata.hdf5")
        if save_path != None:
            autoencoder.save(save_path)
        elif "datapath" in os.environ.keys():
            autoencoder.save(
                os.environ["datapath"]
                + "/"
                + os.environ["experiment_name"]
                + "/NN/autoencoder_newdata.hdf5"
            )

        return autoencoder


class NNutils:
    def scale_log_data_targets(df_train, df_val, df_test):
        y_train = df_train["time"].values.reshape(-1, 1)
        y_val = df_val["time"].values.reshape(-1, 1)
        y_test = df_test["time"].values.reshape(-1, 1)

        y_val_log = np.log(y_val)
        y_train_log = np.log(y_train)
        y_test_log = np.log(y_test)

        y_train_log_min = np.min(y_train_log)
        y_train_min = np.min(y_train)

        y_train_log_max = np.max(y_train_log)
        y_train_max = np.max(y_train)

        print("targets min:{} max: {}".format(y_train_min, y_train_max))
        print(
            "targets in log scale min:{} max: {}".format(
                y_train_log_min, y_train_log_max
            )
        )

        return (
            df_train.drop(columns=["time"]),
            df_val.drop(columns=["time"]),
            df_test.drop(columns=["time"]),
            y_train,
            y_val,
            y_test,
            y_train_log,
            y_val_log,
            y_test_log,
        )

    def printSTDVARMEAN(stastNoAEC, title):
        print(title)
        print("STD")
        print(stastNoAEC.std())
        print("VAR")
        print(stastNoAEC.std() ** 2)
        print("MEAN")
        print(stastNoAEC.mean())

    def normalize_target(y_train_log, y_val_log, y_test_log):
        """
        Normalize data using StandardScaler.

        return scaler object; values of train,val and test sets standarized.
        """
        # Standarización del target
        scaler = StandardScaler()
        y_train_log_std = scaler.fit_transform(y_train_log)
        y_val_log_std = scaler.transform(y_val_log)
        y_test_log_std = scaler.transform(y_test_log)
        return scaler, y_train_log_std, y_val_log_std, y_test_log_std

    def get_metrics(model, scaler, x_data, y_true_data, label_set="Data"):
        y_pred = np.exp(scaler.inverse_transform(model.predict(x_data).reshape(-1, 1)))
        try:
            rmse = np.sqrt(mean_squared_error(y_true_data, y_pred))
            r2 = r2_score(y_true_data, y_pred)
        except ValueError:
            rmse = np.finfo(np.float32).max
            r2 = np.finfo(np.float32).max
        print("RMSE " + label_set, rmse)
        print("R2 SCORE " + label_set, r2)
        return rmse, r2

    def plot_history(history, metrics_list, start_at_epoch=0):
        plt.clf()
        for metric in metrics_list:
            plt.plot(history.history[metric][start_at_epoch:], label=metric)
        plt.legend()
        plt.title("Metrics by epochs(Start from epoch:{})".format(start_at_epoch))
        plt.savefig("plot" + str(time()) + ".png", format="png")
        plt.show()

    def coeff_determination(y_true, y_pred):
        """Coefficient of determination to use in metrics calculated by epochs"""
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    def rmse(y_true, y_pred):
        """RMSE to use in metrics calculated by epochs"""
        return K.exp(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))
