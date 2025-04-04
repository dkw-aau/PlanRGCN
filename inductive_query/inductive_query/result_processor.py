import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
class ResultProcessor:
    """This class should be a general result processing class. For one Prediction class.
    """
    gt_labels = [0,1,2]
    
    def __init__(self, path_to_pred, sep = ',', ground_truth_col='time_cls', pred_col='planrgcn_prediction', id_col='id', ground_truth_list =[0,1,2,],ground_truth_map = {0:"0s-1s", 1: "1s-10s", 2: "10s-∞"}, approach_name="PlanRGCN",apply_cls_func=None):
        """Prepares the result processor.
        In general case, the following methods need to run before result analysis:
            .set_train_pred_ents()
            .set_train_pred_queryIDs()
            .set_train_ent_queryIDs()
            .set_test_pred_ents()
            .set_test_pred_queryIDs()
            .set_test_ent_queryIDs()

        Args:
            path_to_pred (str): Path to file with predictions and ground truth.
            sep (str, optional): seperator used seperate columns in path_to_pred. Defaults to ','.
            ground_truth_col (str, optional): ground truth column name in path_to_pred. Defaults to 'time_cls'.
            pred_col (str, optional): prediction column in path_to_pred. Defaults to 'planrgcn_prediction'.
            id_col (str, optional): queryID column in path_to_pred. Defaults to 'id'.re
            ground_truth_labels (list, optional): The ordering of prediction meanings (e.g., 0-> [0-1], 1 -> [1,10], 2->[10, inf]). Defaults to [0,1,2] for 3 time interval classification.
            ground_truth_map (dict, optional): The ordering of prediction meanings (e.g., 0-> [0-1], 1 -> [1,10], 2->[10, inf]). Defaults to [0,1,2] for 3 time interval classification.
        """
        self.pred_file = path_to_pred #prediction file
        self.sep = sep
        
        self.ground_truth_col = ground_truth_col
        self.pred_col = pred_col
        self.id_col = id_col
        
        self.ground_truth_map = ground_truth_map
        self.ground_truth_list = ground_truth_list
        self.approach_name = approach_name
        
        self.df = pd.read_csv(self.pred_file, sep = self.sep)
        self.apply_cls_func = apply_cls_func
        
        # Code for baseline analysis
        if apply_cls_func != None:
            if 'time' in self.df.columns:
                self.df[self.ground_truth_col] = self.df['time'].apply(apply_cls_func)
            if 'svm_prediction' in self.df.columns:
                self.df[self.pred_col] = self.df['svm_prediction'].apply(apply_cls_func)
            if 'nn_prediction' in self.df.columns:
                self.df[self.pred_col] = self.df['nn_prediction'].apply(apply_cls_func)
            if 'planrgcn_prediction' in self.df.columns:
                self.df[self.pred_col] = self.df['planrgcn_prediction'].apply(apply_cls_func)
        if 'nn_prediction' in self.df.columns:
            self.pred_col = 'nn_prediction'
        elif 'svm_prediction' in self.df.columns:
            self.pred_col = 'svm_prediction'
        elif 'planrgcn_prediction' in self.df.columns:
            self.pred_col = 'planrgcn_prediction'
        
        
        if self.df.iloc[0][self.id_col].startswith("http"):
            self.df[self.id_col] = self.df[self.id_col].apply(lambda x: x[20:])
        self.latex_options = {'decimal':'.','float_format':"%.2f"}
    
    def remove_ids(self, ids):
        self.df = self.df[~self.df[self.id_col].isin(ids)]
        self.df = self.df.reset_index(drop=True)
    
    def retain_ids(self, ids, remove_prefix=0):
        #if len(ids) > 0 and ids[0].startswith('http'):
        ids = [x[remove_prefix:] for x in ids]
        self.df = self.df[self.df[self.id_col].isin(ids)]
        self.df = self.df.reset_index(drop=True)
    
    def retain_path(self, path, sep='\t', id_col='id', remove_prefix=0):
        df = pd.read_csv(path, sep=sep)
        """if remove_lsq != -1:
            ids = [x[remove_lsq:] for x in list(df[id_col])]
        else:"""
        ids = [x for x in list(df[id_col])]
        self.retain_ids(ids, remove_prefix=remove_prefix)
        
    def get_predictions(self):
        return self.df[self.pred_col].to_numpy()
    
    def get_ground_truth(self):
        return self.df[self.ground_truth_col].to_numpy()
        
    
    def check_ids(self, *id):
        """return true if one of the provided id exist in input data
        Returns:
            boolean: _description_
        """
        
        return True if np.sum(self.df[self.id_col].isin([*id])) > 0 else False
    
    def confusion_matrix_raw(self):
        """outputs the confusion matrix as a numpy array

        Returns:
            ndarray: confusion matrix, where rows represent true, and column predictions.
        """
        conf_matrix = confusion_matrix(self.df[self.ground_truth_col], self.df[self.pred_col], labels=ResultProcessor.gt_labels)
        return conf_matrix
    
    def confusion_matrix_to_latex(self, row_percentage=False,name_dict=None, to_latex=True):
        if row_percentage:
            return self.confusion_matrix_to_latex_row_wise(name_dict=name_dict)
        
        conf_matrix = self.confusion_matrix_raw()
        # Load confusion matrix into a pandas DataFrame
        df_confusion = pd.DataFrame(conf_matrix)
        # Rename columns and index if name dictionary is provided
        if name_dict:
            df_confusion = df_confusion.rename(columns=name_dict, index=name_dict)
        # Add row and column names for actual and predicted axes
        df_confusion.columns.name = 'Predicted'
        df_confusion.index.name = 'Actual'
        if not to_latex:
            return df_confusion
        # Convert DataFrame to LaTeX table format
        latex_table = df_confusion.to_latex(multicolumn=True, multicolumn_format='c',**self.latex_options)

        return latex_table

    def confusion_matrix_to_latex_row_wise(self, name_dict=None, return_sums=False, add_sums = False, to_latex=True):
        # Convert confusion matrix to pandas DataFrame
        conf_matrix = self.confusion_matrix_raw()
        conf_matrix, sums = self.compute_percentages_row(conf_matrix)
        df_confusion = pd.DataFrame(conf_matrix)
        
        # Rename columns and index if name dictionary is provided
        if name_dict:
            df_confusion = df_confusion.rename(columns=name_dict, index=name_dict)
        
        
        df_confusion.columns.name = 'Predicted'
        df_confusion.index.name = 'Actual'
        
        # Transpose the DataFrame to switch rows and columns
        #df_confusion = df_confusion.transpose()
        
        # Convert DataFrame to LaTeX table format
        if add_sums:
            df_confusion['\# Total'] = sums.tolist()
        if not to_latex:
            return df_confusion
        latex_table = df_confusion.to_latex(multicolumn=True, multicolumn_format='c',**self.latex_options)
        if return_sums:
            return latex_table,sums
        return latex_table
    
    def confusion_matrix_to_latex_row_wise_sklearn(self, name_dict=None):
        # Convert confusion matrix to pandas DataFrame
        conf_matrix = confusion_matrix(self.df[self.ground_truth_col], self.df[self.pred_col], labels=ResultProcessor.gt_labels,normalize='true')
        df_confusion = pd.DataFrame(conf_matrix)
        
        # Rename columns and index if name dictionary is provided
        if name_dict:
            df_confusion = df_confusion.rename(columns=name_dict, index=name_dict)
        
        
        df_confusion.columns.name = 'Predicted'
        df_confusion.index.name = 'Actual'
        
        # Transpose the DataFrame to switch rows and columns
        #df_confusion = df_confusion.transpose()
        
        # Convert DataFrame to LaTeX table format
        latex_table = df_confusion.to_latex(multicolumn=True, multicolumn_format='c',**self.latex_options)

        return latex_table
    
    
    def get_class_wise_metrics(self):
        predicions = self.get_predictions()
        actual = self.get_ground_truth()
        met_dict = {"Approach": [], "metric_val":[], "Metric": [], "Time Interval": []}
        for met_name,metric_func in zip(['F1', 'Precision', 'Recall'],[f1_score, precision_score, recall_score]):
            met = self.get_class_wise_met(actual,predicions, metric_func)
            for m, k in zip(met, self.ground_truth_list):
                met_dict['Approach'].append(f"{self.approach_name} {self.ground_truth_map[k]}")
                met_dict['metric_val'].append(m)
                met_dict['Metric'].append(met_name)
                met_dict["Time Interval"].append(self.ground_truth_map[k])
        return met_dict
    
    def class_wise_metrics_for_table(self):
        predicions = self.get_predictions()
        actual = self.get_ground_truth()
        met_dict = {"Approach": [], "metric_val":[], "Metric": [], "Time Interval": []}
        met_dict = {}
        met_names = ['F1', 'Precision', 'Recall']
        for met_name,metric_func in zip(met_names,[f1_score, precision_score, recall_score]):
            met = self.get_class_wise_met(actual, predicions, metric_func)
            for m, k in zip(met, self.ground_truth_list):
                try:
                    met_dict[self.ground_truth_map[k]] += f"/{m:.2f}"
                except KeyError:
                    met_dict[self.ground_truth_map[k]] = f"{m:.2f}"

        return met_dict, met_names
        
    def get_class_wise_met(self, actual_values, predicted_values, metric_func
    ):  
        """return the class-wise metric defined by metric_func. If user defined, an additional 'average' keyword arguement is necessary.

        Args:
            actual_values (_type_): _description_
            predicted_values (_type_): _description_
            metric_func (_type_): _description_

        Returns:
            _type_: _description_
        """
        met_val = metric_func(actual_values, predicted_values, average=None)
        d_act = sorted(set(actual_values))
        res_dct = {}
        for idx, i in enumerate(d_act):
            res_dct[i] = met_val[idx]
        res_val = []
        for i in range(np.max(actual_values)+1):
            try:
                res_val.append(res_dct[i])
            except:
                res_val.append(np.nan)
        return res_val
    
    def compute_percentages_row(self, arr):
        # Convert the numpy array to float to avoid integer division
        arr = arr.astype(float)
        
        # Compute the sum of each inner array
        sums = arr.sum(axis=1)
        
        # Compute percentages for each inner array
        percentages = (arr.T / sums * 100).T
        
        return percentages, sums
