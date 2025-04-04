import pandas as pd
import numpy as np
from sklearn.svm import NuSVR
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import warnings

warnings.simplefilter("error")


class SVMTrainerNoPreprocess:
    def __init__(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, resultDir: str
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.resultDir = resultDir
        random.seed(10)

    def trainer(self, C=340, nu=0.10):
        dftable, results_baseline = self.search_hiperparameter_svr(
            self.train, self.val, self.test
        )

        print(dftable)
        exit()
        choice = np.argmin(dftable["rmse_val"])
        C = dftable["C"].iloc[choice]
        nu = dftable["nu"].iloc[choice]
        print(f"Optimal RMSE loss in Val is C={C}, nu={nu}")
        (
            result_table,
            result_baseline_model,
            scaler,
            best_model,
        ) = self.train_bestmodel_svr(C, nu, self.train, self.val, self.test)
        print(result_table)
        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
        ) = self.scale_log_data_targets(self.train, self.val, self.test)

        self.save_svm_prediction(
            self.train,
            x_train,
            scaler,
            best_model,
            self.resultDir + "train_pred.csv",
        )
        self.save_svm_prediction(
            self.val, x_val, scaler, best_model, self.resultDir + "val_pred.csv"
        )
        self.save_svm_prediction(
            self.test, x_test, scaler, best_model, self.resultDir + "test_pred.csv"
        )

    def search_hiperparameter_svr(self, df_train, data_val, data_test):
        result_baseline_model = []
        res_table = []
        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
        ) = self.scale_log_data_targets(df_train, data_val, data_test)
        scaler_yz = StandardScaler()
        scaler_yz = scaler_yz.fit(y_train)
        y_train_scaled_svr = scaler_yz.transform(y_train)
        y_val_scaled_svr = scaler_yz.transform(y_val)
        y_test_scaled_svr = scaler_yz.transform(y_test)

        print("Shape datasets x: {}".format(x_train.shape))
        print("Shape datasets xval: {}".format(x_val.shape))
        print("Shape datasets xtest: {}".format(x_test.shape))
        print("Columns datasets after normalize.", x_train.columns)

        for i in range(1, 10):
            C = 300
            nu = 0.3
            # print(result_baseline_model)
            # Train model
            sv, training_time = self.baseline_svr(
                C, nu, x_train.values, y_train_scaled_svr
            )

            rmse_train, r2_train = self.get_metrics_svr_model(
                sv, x_train, y_train, y_train_scaled_svr, scaler_yz
            )
            rmse_val, r2_val = self.get_metrics_svr_model(
                sv, x_val, y_val, y_val_scaled_svr, scaler_yz
            )
            #     rmse_test, r2_test = get_metrics_svr_model(sv, scalery, x_test, y_test, y_test_hat_svr)

            print("RMSE train: {}, R2 train {}".format(rmse_train, r2_train))
            print("RMSE val: {}, R2 val {}".format(rmse_val, r2_val))
            #     print("MSE test: {}, R2 test:{}".format(mse_svr_test_curr, scores_test_curr))
            res_table.append(
                {
                    "C": C,
                    "nu": nu,
                    "rmse_train": rmse_train,
                    "rmse_val": rmse_val,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                    "training_time": training_time,
                }
            )
            result_baseline_model.append(sv)
        return pd.DataFrame.from_dict(res_table), result_baseline_model

    def train_bestmodel_svr(self, C, nu, df_train, data_val, data_test):
        result_baseline_model = []
        min_rmse = 100000000000000
        results = []
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
        ) = self.scale_log_data_targets(df_train, data_val, data_test)
        print("Shape datasets x: {}".format(x_train.shape))
        print("Shape datasets xval: {}".format(x_val.shape))
        print("Shape datasets xtest: {}".format(x_test.shape))
        print("Columns datasets after normalize.", x_train.columns)
        # scale target using StandarSacaler
        scalery, y_train_log_std, y_val_log_std, y_test_log_std = self.normalize_target(
            y_train_log, y_val_log, y_test_log
        )

        for i in range(1, 10):
            # result_baseline_model
            # Train model
            sv, training_time = self.baseline_svr(
                C, nu, x_train.values, y_train_log_std
            )

            rmse_train, r2_train = self.get_metrics_svr_model(
                sv, scalery, x_train.values, y_train, y_train_log_std
            )
            rmse_val, r2_val = self.get_metrics_svr_model(
                sv, scalery, x_val.values, y_val, y_val_log_std
            )
            rmse_test, r2_test = self.get_metrics_svr_model(
                sv, scalery, x_test.values, y_test, y_test_log_std
            )

            print("RMSE train: {}, R2 train {}".format(rmse_train, r2_train))
            print("RMSE val: {}, R2 val {}".format(rmse_val, r2_val))
            print("RMSE test: {}, R2 test:{}".format(rmse_test, r2_test))
            results.append(
                {
                    "C": C,
                    "nu": nu,
                    "rmse_train": rmse_train,
                    "rmse_val": rmse_val,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                    "training_time": training_time,
                }
            )

            if rmse_test < min_rmse:
                best_model = sv
                min_rmse = rmse_test
            result_baseline_model.append(sv)
        return (
            pd.DataFrame.from_dict(results),
            result_baseline_model,
            scalery,
            best_model,
        )

    def scale_log_data_targets(self, df_train, df_val, df_test):
        y_train = df_train["time"].values.reshape(-1, 1)
        y_val = df_val["time"].values.reshape(-1, 1)
        y_test = df_test["time"].values.reshape(-1, 1)

        # return df_train.drop(columns=['time','id']), df_val.drop(columns=['time','id']), df_test.drop(columns=['time','id']) , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log
        """drop_column = []
        for col in df_train.columns:
            print(col)
            if col in ['time','execTime','id','queryID']:
                drop_column.append(col)"""
        #'joins_count', 'filter_uri', 'filter_num', 'filter_literal', are filtered because they should be additional features.
        retain = [
            "triple",
            "bgp",
            "leftjoin",
            "union",
            "filter",
            "graph",
            "extend",
            "minus",
            "order",
            "project",
            "distinct",
            "group",
            "slice",
            "treesize",
        ]
        # adds ged features
        for x in df_train.columns:
            if x.startswith("cls_"):
                retain.append(x)
        return (
            df_train[retain],
            df_val[retain],
            df_test[retain],
            y_train,
            y_val,
            y_test,
        )

    def normalize_target(self, y_train_log, y_val_log, y_test_log):
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

    def save_svm_prediction(self, df, xs, scaler, model, path_to_save):
        y_train_hat_svr = scaler.inverse_transform(
            np.exp(model.predict(xs.values).reshape(-1, 1))
        )
        # y_train_hat_svr = scaler.inverse_transform( model.predict(xs).reshape(-1, 1))

        df = pd.DataFrame(
            {
                "id": df["id"],
                "time": df["time"],
                "svm_prediction": y_train_hat_svr.flatten(),
            }
        )
        df.to_csv(path_to_save)

    def baseline_svr(self, C, nu, Xdata, Ydata):
        sv = NuSVR(C=C, nu=nu)

        t0 = time()
        print("before train: Init time: {}".format(round(t0, 3)))
        Ydata = np.ravel(Ydata)
        sv.fit(Xdata, Ydata)
        t1 = time()

        print("after train, finish time: {}".format(round(t1, 3)))
        print("training time {}", format(round(t1 - t0, 3)))
        return [sv, round(t1 - t0, 3)]

    def get_metrics_svr_model(self, sv, x_train, y_train, y_train_scaled_svr, scaler):
        y_train_hat_svr = scaler.inverse_transform(sv.predict(x_train).reshape(-1, 1))

        # MSE for valid
        # mse_svr_curr = np.sqrt(mean_squared_error( y_train_hat_svr,y_train))
        mse_svr_curr = np.sqrt(mean_squared_error(y_train_hat_svr, y_train))

        scores_train_curr = sv.score(x_train, y_train_scaled_svr)

        return mse_svr_curr, scores_train_curr


class SVMTrainer:
    def __init__(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, resultDir: str
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.resultDir = resultDir
        random.seed(10)

    def trainer(self, C=340, nu=0.10):
        dftable, results_baseline = self.search_hiperparameter_svr(
            self.train, self.val, self.test
        )
        print(dftable)
        choice = np.argmin(dftable["rmse_val"])
        C = dftable["C"].iloc[choice]
        nu = dftable["nu"].iloc[choice]
        self.C = C
        self.nu = nu
        print(f"Optimal RMSE loss in Val is C={C}, nu={nu}")
        (
            result_table,
            result_baseline_model,
            scaler,
            best_model,
        ) = self.train_bestmodel_svr(C, nu, self.train, self.val, self.test)
        self.best_model = best_model
        self.result_table = result_table
        self.result_baseline_model = result_baseline_model
        self.scaler = scaler
        print(result_table)
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
        ) = self.scale_log_data_targets(self.train, self.val, self.test)

        self.save_svm_prediction(
            self.train,
            x_train,
            scaler,
            best_model,
            self.resultDir + "train_pred.csv",
        )
        self.save_svm_prediction(
            self.val, x_val, scaler, best_model, self.resultDir + "val_pred.csv"
        )
        self.save_svm_prediction(
            self.test, x_test, scaler, best_model, self.resultDir + "test_pred.csv"
        )
        
    def predict_trained(self, test= None, output_path= None):
        if output_path == None:
            print('Please valide output_path')
            exit()
        C = self.C
        nu = self.nu
        
        best_model = self.best_model 
        result_table = self.result_table
        result_baseline_model = self.result_baseline_model
        scaler = self.scaler
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
        ) = self.scale_log_data_targets(self.train, self.val, test)

        
        self.save_svm_prediction(
            test, x_test, scaler, best_model, output_path
        )

    def search_hiperparameter_svr(self, df_train, data_val, data_test):
        result_baseline_model = []
        res_table = []
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
        ) = self.scale_log_data_targets(df_train, data_val, data_test)
        print("Shape datasets x: {}".format(x_train.shape))
        print("Shape datasets xval: {}".format(x_val.shape))
        print("Shape datasets xtest: {}".format(x_test.shape))
        print("Columns datasets after normalize.", x_train.columns)
        # scale target using StandarSacaler
        scalery, y_train_log_std, y_val_log_std, y_test_log_std = self.normalize_target(
            y_train_log, y_val_log, y_test_log
        )

        for i in range(1, 10):
            C = random.randrange(100, 350, 20)
            nu = random.randrange(10, 50, 5) / 100
            # print(result_baseline_model)
            # Train model
            sv, training_time = self.baseline_svr(
                C, nu, x_train.values, y_train_log_std
            )

            rmse_train, r2_train = self.get_metrics_svr_model(
                sv, scalery, x_train.values, y_train, y_train_log_std
            )
            rmse_val, r2_val = self.get_metrics_svr_model(
                sv, scalery, x_val.values, y_val, y_val_log_std
            )
            #     rmse_test, r2_test = get_metrics_svr_model(sv, scalery, x_test, y_test, y_test_hat_svr)

            print("RMSE train: {}, R2 train {}".format(rmse_train, r2_train))
            print("RMSE val: {}, R2 val {}".format(rmse_val, r2_val))
            #     print("MSE test: {}, R2 test:{}".format(mse_svr_test_curr, scores_test_curr))
            res_table.append(
                {
                    "C": C,
                    "nu": nu,
                    "rmse_train": rmse_train,
                    "rmse_val": rmse_val,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                    "training_time": training_time,
                }
            )
            result_baseline_model.append(sv)
        return pd.DataFrame.from_dict(res_table), result_baseline_model

    def train_bestmodel_svr(self, C, nu, df_train, data_val, data_test):
        result_baseline_model = []
        min_rmse = 100000000000000
        results = []
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
        ) = self.scale_log_data_targets(df_train, data_val, data_test)
        print("Shape datasets x: {}".format(x_train.shape))
        print("Shape datasets xval: {}".format(x_val.shape))
        print("Shape datasets xtest: {}".format(x_test.shape))
        print("Columns datasets after normalize.", x_train.columns)
        # scale target using StandarSacaler
        scalery, y_train_log_std, y_val_log_std, y_test_log_std = self.normalize_target(
            y_train_log, y_val_log, y_test_log
        )

        for i in range(1, 10):
            # result_baseline_model
            # Train model
            sv, training_time = self.baseline_svr(
                C, nu, x_train.values, y_train_log_std
            )

            rmse_train, r2_train = self.get_metrics_svr_model(
                sv, scalery, x_train.values, y_train, y_train_log_std
            )
            rmse_val, r2_val = self.get_metrics_svr_model(
                sv, scalery, x_val.values, y_val, y_val_log_std
            )
            rmse_test, r2_test = self.get_metrics_svr_model(
                sv, scalery, x_test.values, y_test, y_test_log_std
            )

            print("RMSE train: {}, R2 train {}".format(rmse_train, r2_train))
            print("RMSE val: {}, R2 val {}".format(rmse_val, r2_val))
            print("RMSE test: {}, R2 test:{}".format(rmse_test, r2_test))
            results.append(
                {
                    "C": C,
                    "nu": nu,
                    "rmse_train": rmse_train,
                    "rmse_val": rmse_val,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                    "training_time": training_time,
                }
            )

            if rmse_test < min_rmse:
                best_model = sv
                min_rmse = rmse_test
            result_baseline_model.append(sv)
        return (
            pd.DataFrame.from_dict(results),
            result_baseline_model,
            scalery,
            best_model,
        )

    def scale_log_data_targets(self, df_train, df_val, df_test):
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

        # return df_train.drop(columns=['time','id']), df_val.drop(columns=['time','id']), df_test.drop(columns=['time','id']) , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log
        """drop_column = []
        for col in df_train.columns:
            print(col)
            if col in ['time','execTime','id','queryID']:
                drop_column.append(col)"""
        #'joins_count', 'filter_uri', 'filter_num', 'filter_literal', are filtered because they should be additional features.
        retain = [
            "triple",
            "bgp",
            "leftjoin",
            "union",
            "filter",
            "graph",
            "extend",
            "minus",
            "order",
            "project",
            "distinct",
            "group",
            "slice",
            "treesize",
        ]
        # adds ged features
        for x in df_train.columns:
            if x.startswith("cls_"):
                retain.append(x)
        return (
            df_train[retain],
            df_val[retain],
            df_test[retain],
            y_train,
            y_val,
            y_test,
            y_train_log,
            y_val_log,
            y_test_log,
        )

    def normalize_target(self, y_train_log, y_val_log, y_test_log):
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

    def save_svm_prediction(self, df, xs, scaler, model, path_to_save):
        start = time()
        y_train_hat_svr = scaler.inverse_transform(
            np.exp(model.predict(xs.values).reshape(-1, 1))
        )
        dur = time() - start
        dur = dur/len(df)
        # y_train_hat_svr = scaler.inverse_transform( model.predict(xs).reshape(-1, 1))

        df = pd.DataFrame(
            {
                "id": df["id"],
                "time": df["time"],
                "svm_prediction": y_train_hat_svr.flatten(),
            }
        )
        df['inference'] =dur
        df.to_csv(path_to_save)

    def baseline_svr(self, C, nu, Xdata, Ydata):
        sv = NuSVR(C=C, nu=nu)

        t0 = time()
        print("before train: Init time: {}".format(round(t0, 3)))
        Ydata = Ydata.squeeze()
        sv.fit(Xdata, Ydata)
        t1 = time()

        print("after train, finish time: {}".format(round(t1, 3)))
        print("training time {}", format(round(t1 - t0, 3)))
        return [sv, round(t1 - t0, 3)]

    def get_metrics_svr_model(self, sv, scalery, x_train, y_train, y_log_std):
        y_train_hat_svr = scalery.inverse_transform(sv.predict(x_train).reshape(-1, 1))

        # MSE for valid
        mse_svr_curr = np.sqrt(mean_squared_error(y_train, y_train_hat_svr))

        scores_train_curr = sv.score(x_train, y_log_std)

        return mse_svr_curr, scores_train_curr
