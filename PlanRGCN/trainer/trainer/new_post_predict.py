from collections import Counter

import numpy as np
import pandas as pd
from pandas import MultiIndex
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import argparse, ast



import os

from q_gen.util import Utility

os.environ['QG_JAR'] = '/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
from graph_construction.jar_utils import *


class QPPResultProcessor:
    gt_labels = [0,1,2]
    def __init__(self,
                 obj_fil=None,
                 dataset = None,
                 exp_type="all",
                 test_sampled_file=None, # required for PP filter because of queryString
                 train_sampled_file = None,  # only used for unseen analysis
                 val_sampled_file = None,  # only used for unseen analysis
                 ):


        # Load correct classification snapper and corresponding data
        self.true_counts = {}
        self.methods = []
        self.dataset = dataset
        self.exp_type = exp_type

        self.test_sampled_file=test_sampled_file
        self.train_sampled_file = train_sampled_file
        self.val_sampled_file = val_sampled_file

        if not (obj_fil == None or obj_fil == "None"):
            exec(open(obj_fil).read(), globals())
            self.label_map = {}
            self.label_index =[]
            if 'thresholds' in globals():
                global thresholds
                global cls_func
                for i in range(len(thresholds)-1):
                    self.label_map[i] = f"({thresholds[i]:.4f};{thresholds[i+1]:.4f}]"
                    self.label_index.append(f"({thresholds[i]:.4f};{thresholds[i+1]:.4f}]")
        if not 'cls_func' in globals() or (cls_func == None):
            from graph_construction.query_graph import snap_lat2onehotv2
            QPPResultProcessor.cls_func = snap_lat2onehotv2
        else:
            QPPResultProcessor.cls_func = cls_func
        self.interval = 3
        QPPResultProcessor.gt_labels = [i for i in range(self.interval)]

        #Metric that we want to compare multiple datasets on:
        self.conf_matrix_entries = [] # to make augmented conf_matrix
        #We need to some for class wise F1, precisio, recall
        self.method_results = {}

    def evaluate_dataset(self, path_to_pred="path_to_pred",
                         sep = ',',
                         ground_truth_col='time_cls',
                         pred_col=None,
                         id_col='id',
                         approach_name="PlanRGCN",
                         reg_to_cls = True
                         ):

        self.pred_file = path_to_pred  # prediction file
        self.sep = sep

        self.ground_truth_col = ground_truth_col
        self.pred_col = pred_col
        self.id_col = id_col

        self.methods.append(approach_name)


        self.df = pd.read_csv(self.pred_file, sep=self.sep)
        match self.exp_type:
            case "completly_unseen":
                assert self.test_sampled_file != None
                assert self.val_sampled_file != None
                assert self.train_sampled_file != None
                def normalize_id(x):
                    if x.startswith('http'):
                        return x[20:]
                    return x

                if not 'id' in self.df.columns and 'queryID' in self.df.columns:
                    self.df['id'] = self.df['queryID'].apply(normalize_id)
                else:
                    self.df['id'] = self.df['id'].apply(normalize_id)

                l_df = len(self.df)
                q_df = pd.read_csv(self.test_sampled_file, sep='\t')[['queryID', 'queryString']].rename(
                    columns={'queryID': 'id'})
                q_df['id'] = q_df['id'].apply(normalize_id)

                self.df = self.df.merge(q_df, how='inner', on='id')
                assert len(self.df) == l_df
                non_test_terms = set()
                train_df = pd.read_csv(self.train_sampled_file, sep='\t')
                val_df = pd.read_csv(self.train_sampled_file, sep='\t')
                for idx, row in train_df.iterrows():
                    try:
                        ents, rels = get_ent_rel(row['queryString'])
                        non_test_terms.update(ents)
                        non_test_terms.update(rels)
                    except Exception:
                        continue

                """for idx, row in val_df.iterrows():
                    try:
                        ents, rels = get_ent_rel(row['queryString'])
                        non_test_terms.update(ents)
                        non_test_terms.update(rels)
                    except Exception:
                        continue"""

                complete_unseen_idx = []
                for idx, row in self.df.iterrows():
                        ents, rels = get_ent_rel(row['queryString'])
                        is_completely_unseen = True
                        for e in ents:
                            if e in non_test_terms:
                                is_completely_unseen = False
                        for r in rels:
                            if r in non_test_terms:
                                is_completely_unseen = False
                        if is_completely_unseen:
                            complete_unseen_idx.append(row['id'])

                self.df = self.df[self.df['id'].isin(complete_unseen_idx)].copy()
            case "unseen_entity":
                ... # implement filter functionality on unseen entity in test set
            case "unseen_relation":
                ... # implement filter functionality on unseen relation in test set
            case "PP":
                self.df = self.df
                def normalize_id(x):
                    if x.startswith('http'):
                        return x[20:]
                    return x

                if not 'id' in self.df.columns and 'queryID' in self.df.columns:
                    self.df['id'] = self.df['queryID'].apply(normalize_id)
                else:
                    self.df['id'] = self.df['id'].apply(normalize_id)

                assert self.test_sampled_file != None
                l_df = len(self.df)
                q_df = pd.read_csv(self.test_sampled_file, sep='\t')[['queryID', 'queryString']].rename(columns={'queryID':'id'})
                q_df['id'] =q_df['id'].apply(normalize_id)
                self.df = self.df.merge(q_df, how='inner', on='id')
                assert l_df>= len(self.df)

                #Filtered for only queries with PP
                #self.df['queryString'].apply(lambda x: check_PP(x))
                def pp_filter(x):
                    try:
                        return check_PP(x)
                    except Exception:
                        return False

                self.df = self.df[self.df['queryString'].apply(lambda x: pp_filter(x))].copy()

            case "seen_PP":
                self.df = self.df

                def normalize_id(x):
                    if x.startswith('http'):
                        return x[20:]
                    return x

                if not 'id' in self.df.columns and 'queryID' in self.df.columns:
                    self.df['id'] = self.df['queryID'].apply(normalize_id)
                else:
                    self.df['id'] = self.df['id'].apply(normalize_id)
                assert self.test_sampled_file != None
                l_df = len(self.df)
                q_df = pd.read_csv(self.test_sampled_file, sep='\t')[['queryID', 'queryString']].rename(
                    columns={'queryID': 'id'})
                q_df['id'] = q_df['id'].apply(normalize_id)
                self.df = self.df.merge(q_df, how='inner', on='id')
                assert l_df >= len(self.df)

                # Filtered for only queries with PP
                # self.df['queryString'].apply(lambda x: check_PP(x))
                def pp_filter(x):
                    try:
                        return check_PP(x)
                    except Exception:
                        return False

                self.df = self.df[self.df['queryString'].apply(lambda x: pp_filter(x))].copy()
                train_df = pd.read_csv(self.train_sampled_file, sep='\t')
                train_rels, train_ents = Utility.get_ent_rels_from_train(train_df)
                def checkIfSeen(query, train_rels, train_ents):
                    ents, rels = Utility.get_ent_rel(query)
                    for e in ents:
                        if e not in train_ents:
                            return False

                    for r in rels:
                        if r not in train_rels:
                            return False

                    return True

                self.df = self.df[self.df['queryString'].apply(lambda x: checkIfSeen(x, train_rels, train_ents))].copy()




            case _:
                ... #in default case make no filtering
        self.thresholds = [0,1,10,900] # old defualts



        # Code for baseline analysis
        if reg_to_cls and (('svm_prediction' in self.df.columns) or ('nn_prediction' in self.df.columns)):
            if 'time' in self.df.columns:
                self.df[self.ground_truth_col] = self.df['time'].apply(lambda x: np.argmax(QPPResultProcessor.cls_func(x)))
            if 'svm_prediction' in self.df.columns:
                self.df[self.pred_col] = self.df['svm_prediction'].apply(lambda x: np.argmax(QPPResultProcessor.cls_func(x)))
            if 'nn_prediction' in self.df.columns:
                self.df[self.pred_col] = self.df['nn_prediction'].apply(lambda x: np.argmax(QPPResultProcessor.cls_func(x)))

        if pred_col != None:
            self.pred_col = pred_col
        elif 'nn_prediction' in self.df.columns:
            self.pred_col = 'nn_prediction'
        elif 'svm_prediction' in self.df.columns:
            self.pred_col = 'svm_prediction'
        elif 'planrgcn_prediction' in self.df.columns:
            self.pred_col = 'planrgcn_prediction'

        self.latex_options = {'decimal': '.', 'float_format': "%.2f"}

        self.conf_matrix_entries.extend(self.get_confusion_matrix_tuples())
        self.true_counts[approach_name] = dict(Counter(self.df[self.ground_truth_col]))
        self.method_results[approach_name] = self.compute_aggregated_metrics()


    def process_results(self, add_symbol='\%', version='VLDB'):
        assert len(self.conf_matrix_entries) > 0
        c_df = self.entry_to_conf_df(self.conf_matrix_entries, version=version)
        latex_options = {'decimal': '.', 'float_format': lambda x: str(round(x*100, 1))+add_symbol}
        latex_table = c_df.to_latex(multicolumn=True, multicolumn_format='c', **latex_options)
        aggre_latex_table = self.process_aggregated_metrics()
        return latex_table, aggre_latex_table



    def get_predictions(self):
        return self.df[self.pred_col].to_numpy()

    def get_ground_truth(self):
        return self.df[self.ground_truth_col].to_numpy()

    def confusion_matrix_row_wise_sklearn(self):

        return pd.DataFrame(confusion_matrix(self.df[self.ground_truth_col], self.df[self.pred_col],
                                      labels=QPPResultProcessor.gt_labels, normalize='true')).rename(
            columns=self.label_map, index=self.label_map)

    def get_confusion_matrix_tuples(self):
        """

        @return: list(tuple), a list of entries.
        An entry is a tuple, of :
            - method name
            - ground truth interval
            - predicted interval
            - number of queriees or row-wise percentage depending on use case
        """
        conf = self.confusion_matrix_row_wise_sklearn()
        entries = []
        for act_idx in conf.index:
            for pred_idx in conf.index:
                entry = (self.methods[-1], act_idx, pred_idx, conf.loc[act_idx, pred_idx])
                entries.append(entry)
        return entries


    def entry_to_conf_df(self, entries, version='VLDB'):
        if version == 'VLDB':
            return self.entry_to_conf_df_VLDBrev(entries)
        else:
            index_ordering = []
            for int in self.label_index:
                for m in self.methods:
                    index_ordering.append((int, m))
            index = MultiIndex.from_tuples(index_ordering)

            return pd.DataFrame(entries,
                         columns=["Method", "ActualInt", "PredInt", "Value"])\
                .set_index(["ActualInt"])\
                .pivot(columns=["PredInt", "Method"], values='Value').reindex(index=self.label_index, columns=index)

    def entry_to_conf_df_VLDBrev(self, entries):
        index_ordering = []
        for m in self.methods:
            for int in self.label_index:
                index_ordering.append((m,int))
        index = MultiIndex.from_tuples(index_ordering)

        return pd.DataFrame(entries,
                     columns=["Method", "ActualInt", "PredInt", "Value"])\
            .set_index(["ActualInt"])\
            .pivot(columns=["Method", "PredInt"], values='Value').reindex(index=self.label_index, columns=index)
    @staticmethod
    def latexify_cf(df_confusion):
        df_confusion.columns.name = 'Predicted'
        df_confusion.index.name = 'Actual'
        # Transpose the DataFrame to switch rows and columns
        # df_confusion = df_confusion.transpose()

        # Convert DataFrame to LaTeX table format
        latex_options = {'decimal': '.', 'float_format': "%.1f"}
        latex_table = df_confusion.to_latex(multicolumn=True, multicolumn_format='c', **latex_options)

        return latex_table
    def compute_aggregated_metrics(self):
        t = self.df[self.ground_truth_col]
        p = self.df[self.pred_col]
        labels = QPPResultProcessor.gt_labels
        acc = accuracy_score(y_true=t, y_pred=p)
        f1_micro = f1_score(y_true=t, y_pred=p, labels=labels, average='micro')
        f1_macro = f1_score(y_true=t, y_pred=p, labels=labels, average='macro')

        precision_micro = precision_score(y_true=t, y_pred=p, labels=labels, average='micro')
        precision_macro = precision_score(y_true=t, y_pred=p, labels=labels, average='macro')

        recall_micro = recall_score(y_true=t, y_pred=p, labels=labels, average='micro')
        recall_macro = recall_score(y_true=t, y_pred=p, labels=labels, average='macro')
        return {'Accuracy': acc, 'F1-micro': f1_micro, 'F1-macro':f1_macro, 'precision micro': precision_micro, 'precision macro': precision_macro, 'recall micro': recall_micro, 'recall macro': recall_macro
                }
    def process_aggregated_metrics(self):
        re_df = pd.DataFrame(self.method_results).T
        latex_table = re_df.to_latex(float_format="%.3f", caption="Aggregated Metrics Table",
                                  label="tab:performance_metrics")
        return latex_table





def parse_args():
    parser = argparse.ArgumentParser(description="Process input arguments for prediction analysis.")

    # Prediction file (TSV file)
    parser.add_argument(
        '--pred_file',
        type=str,
        required=True,
        help="Path to the prediction file (TSV format)."
    )

    # Experiment name
    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help="Name of the experiment."
    )

    # Separator character
    parser.add_argument(
        '--sep',
        type=str,
        default='\t',
        help="Separator character used in the TSV file (default is tab '\\t')."
    )

    # Ground truth column name
    parser.add_argument(
        '--gt_col',
        type=str,
        required=True,
        help="Name of the ground truth column."
    )

    # Prediction column name
    parser.add_argument(
        '--pred_col',
        type=str,
        required=True,
        help="Name of the prediction column."
    )

    # ID column name
    parser.add_argument(
        '--id_col',
        type=str,
        required=True,
        help="Name of the ID column."
    )

    # Ground truth map (a dictionary)
    parser.add_argument(
        '--ground_truth_map',
        type=str,
        required=True,
        help="Mapping of ground truth values as a dictionary (e.g., \"{'label1': 1, 'label2': 0}\")."
    )

    # Objective file (.py file)
    parser.add_argument(
        '--objective_file',
        type=str,
        required=True,
        help="Path to the objective file (.py file)."
    )

    args = parser.parse_args()

    # Convert ground_truth_map argument from string to dictionary
    args.ground_truth_map = ast.literal_eval(args.ground_truth_map)

    return args


if __name__ == "__main__":
    """args = parse_args()

    # Example of how to access the parsed arguments
    print(f"Prediction file: {args.prediction_file}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Separator: {args.sep}")
    print(f"Ground truth column: {args.ground_truth_col}")
    print(f"Prediction column: {args.pred_col}")
    print(f"ID column: {args.id_col}")
    print(f"Ground truth map: {args.ground_truth_map}")
    print(f"Objective file: {args.objective_file}")"""

    # Add your main logic here using the parsed arguments

    print('Test')
    p = QPPResultProcessor(obj_fil='/data/DBpediaV2/plan01/objective.py',
                           dataset="DBpedia",
                           exp_type='PP',
                           test_sampled_file='/data/DBpediaV2/test_sampled.tsv',
                           train_sampled_file='/data/DBpediaV2/train_sampled.tsv',
                           val_sampled_file='/data/DBpediaV2/val_sampled.tsv')

    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/plan01/test_pred.csv",
                           sep = ',',
                           ground_truth_col='time_cls',
                           pred_col='planrgcn_prediction',
                           id_col='id',
                           approach_name="P")

    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/nn/k25/nn_test_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='nn_prediction',
                       id_col='id',
                       approach_name="NN")

    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/svm/test_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='svm_prediction',
                       id_col='id',
                       approach_name="SVM")

    print(p.process_results())


    exit()
    """
    print('val')
    p = QPPResultProcessor(obj_fil='/data/DBpediaV2/plan01/objective.py', dataset="DBpedia")
    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/plan01/val_pred.csv",
                           sep = ',',
                           ground_truth_col='time_cls',
                           pred_col='planrgcn_prediction',
                           id_col='id',
                           approach_name="PlanRGCN")

    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/svm/val_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='svm_prediction',
                       id_col='id',
                       approach_name="SVM")
    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/nn/k25/nn_val_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='nn_prediction',
                       id_col='id',
                       approach_name="NN")
    print(p.process_results())
    print('train')
    p = QPPResultProcessor(obj_fil='/data/DBpediaV2/plan01/objective.py', dataset="DBpedia")

    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/svm/train_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='svm_prediction',
                       id_col='id',
                       approach_name="SVM")
    p.evaluate_dataset(path_to_pred="/data/DBpediaV2/nn/k25/nn_train_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='nn_prediction',
                       id_col='id',
                       approach_name="NN")
    print(p.process_results())
    exit()
    p.evaluate_dataset(path_to_pred="/data/wikidataV2/plan01/test_pred.csv",
                       sep=',',
                       ground_truth_col='time_cls',
                       pred_col='planrgcn_prediction',
                       id_col='id',
                       approach_name="NN")"""