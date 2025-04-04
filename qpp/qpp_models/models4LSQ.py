import os
import pandas as pd
import argparse
from svm import load_dataset_preserving_split,load_cluster_file, search_hiperparameter_svr
from nn import process_extra, normalizaAlgebra, joinAlgebraGPM, \
printSTDVARMEAN, save_prediction, executar
from sklearn.svm import NuSVR 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from twoStepSvm import *
from util import get_std_data_cols
def load_data(paths= None):
    if paths == None:
        paths = {
            'train': '',
            'test' : '',
            'val' : ''
        }
    train = pd.read_csv(paths['train'], sep='\t', engine='python')
    test = pd.read_csv(paths['test'], sep='\t', engine='python')
    val = pd.read_csv(paths['val'], sep='\t', engine='python')

def train_svm(train, val, test, cols, seed=42):
    np.random.seed(seed)
    #svm = NuSVR(nu=0.5,C=1, kernel='rbf')
    regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.5, kernel='rbf'))
    regr.fit(train[[cols]].values(),train['duration'])

    


def print_unique_stat(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    print(f"Unique Stats:\n \tTrain: {len(train[train.duplicated(keep=False)])}\n\tVal: {len(val[val.duplicated(keep=False)])}\n\tTest: {len(test[test.duplicated(keep=False)])}")

def process_svm_lsq_data(args):
    train_graph = load_cluster_file(args.data_path+'train_graph.txt')
    test_graph = load_cluster_file(args.data_path+'test_graph.txt')
    val_graph = load_cluster_file(args.data_path+'val_graph.txt')
    print(len(train_graph), len(test_graph), len(val_graph))
    print(len(train_graph['id'].unique()), len(test_graph['id'].unique()), len(val_graph['id'].unique()))
    print_unique_stat(train_graph,val_graph,test_graph)

    train_df = pd.read_csv(args.data_path+'train.tsv',sep='\t')
    val_df = pd.read_csv(args.data_path+'val.tsv',sep='\t')
    test_df = pd.read_csv(args.data_path+'test.tsv',sep='\t')
    print_unique_stat(train_df, val_df, test_df)


def process_svm_data(args):
    #TODO:Not all queries have these features
    train_graph = load_cluster_file(args.data_path+'train_graph.txt')
    test_graph = load_cluster_file(args.data_path+'test_graph.txt')
    val_graph = load_cluster_file(args.data_path+'val_graph.txt')
    #l_tr,l_v,_lte = len(train_graph), len(test_graph), len(val_graph)
    #print(len(train_graph))
    print(f"Unique {len(train_graph['id'].unique())} of {len(train_graph)}")
    print(f"Unique {len(test_graph['id'].unique())} of {len(test_graph)}")
    print(f"Unique {len(val_graph['id'].unique())} of {len(val_graph)}")
    train_graph = train_graph.drop_duplicates()
    test_graph = test_graph.drop_duplicates()
    val_graph = val_graph.drop_duplicates()
    #print(len(train_graph))

    #print(f"Dublicartes in graph train: {len(train_graph['id'].unique())-len(train_graph['id'])}")
    
    #temporily renaming these files
    #train_df = pd.read_csv(args.data_path+'train.tsv',sep='\t').drop_duplicates()
    train_df = pd.read_csv(args.data_path+'train_sampled.tsv',sep='\t').drop_duplicates()
    #val_df = pd.read_csv(args.data_path+'val.tsv',sep='\t').drop_duplicates()
    val_df = pd.read_csv(args.data_path+'val_sampled.tsv',sep='\t').drop_duplicates()
    #test_df = pd.read_csv(args.data_path+'test.tsv',sep='\t').drop_duplicates()
    test_df = pd.read_csv(args.data_path+'test_sampled.tsv',sep='\t').drop_duplicates()
    len_train = len(train_df)
    
    print(len(train_df), len(test_df), len(val_df))
    #print(train_graph.columns)
    #print(train_df.columns)
    
    train_df = train_df.merge(train_graph, left_on='queryID', right_on='id', how='left')
    
    print(f"Differnece is {len_train-len(train_df)}")
    
    
    val_df = val_df.merge(val_graph, left_on='queryID', right_on='id', how='left')
    test_df = test_df.merge(test_graph, left_on='queryID', right_on='id', how='left')
    print(len(train_df), len(test_df), len(val_df))
    return train_df,val_df,test_df

def svm_pre_process(train, val, test):
    
    scaled_df_train, scaled_df_val, scaled_df_test, scaler = normalizaAlgebra(train, val, test, returnScaler=True)
    col_gpm = ['duration','cls_0', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_6', 'cls_7', 'cls_8', 'cls_9', 'cls_10', 'cls_11', 'cls_12',
        'cls_13', 'cls_14', 'cls_15', 'cls_16', 'cls_17', 'cls_18', 'cls_19', 'cls_20', 'cls_21', 'cls_22', 'cls_23', 'cls_24']
    x_train, x_val, x_test = joinAlgebraGPM(scaled_df_train, scaled_df_val, scaled_df_test, train[col_gpm], val[col_gpm], test[col_gpm])
    return x_train,x_val,x_test


def process_nn_data(args):    
    train_df, val_df, test_df = process_svm_data(args)
    
    data_tpf = pd.read_csv(args.data_path + 'extra/extra.csv', sep="\t") # delimiter ="\t"
    col_to_drop = []
    for col in ["Unnamed: 4",'execTime', 'duration']:
        if col in data_tpf.columns:
            col_to_drop.append(col)
    
    data_tpf = data_tpf.drop(columns=col_to_drop)
    data_tpf_clean = process_extra(data_tpf)
    data_tpf_clean = data_tpf_clean.drop_duplicates()
    
    X_train_extended = train_df.merge( data_tpf_clean, left_on='queryID', right_on='id', how='left')
    X_val_extended = val_df.merge( data_tpf_clean, left_on='queryID', right_on='id', how='left')
    X_test_extended = test_df.merge(data_tpf_clean , left_on='queryID', right_on='id', how='left')
    
    
    X_train_extended['id'] = X_train_extended['queryID']
    X_val_extended['id'] = X_val_extended['queryID']
    X_test_extended['id'] = X_test_extended['queryID']

    X_train_extended = X_train_extended.drop(columns=['id_y','id_x','queryID' ])
    X_val_extended = X_val_extended.drop(columns=['id_y','id_x','queryID' ])
    X_test_extended = X_test_extended.drop(columns=['id_y','id_x','queryID' ])
    
    X_train_extended = X_train_extended.set_index('id')
    X_val_extended = X_val_extended.set_index('id')
    X_test_extended = X_test_extended.set_index('id')
        ###
    X_train_extended = X_train_extended.drop(columns=['join'])
    X_val_extended = X_val_extended.drop(columns=['join'])
    X_test_extended = X_test_extended.drop(columns=['join'])

    scaled_df_train, scaled_df_val, scaled_df_test, scaler = normalizaAlgebra(X_train_extended, X_val_extended, X_test_extended, returnScaler=True)
    col_gpm = [args.latency_col,'cls_0', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_6', 'cls_7', 'cls_8', 'cls_9', 'cls_10', 'cls_11', 'cls_12',
        'cls_13', 'cls_14', 'cls_15', 'cls_16', 'cls_17', 'cls_18', 'cls_19', 'cls_20', 'cls_21', 'cls_22', 'cls_23', 'cls_24']
    t = []
    for x in col_gpm:
        if x in X_train_extended.columns:
            t.append(x)
    col_gpm = t
    x_train, x_val, x_test = joinAlgebraGPM(scaled_df_train, scaled_df_val, scaled_df_test, X_train_extended[col_gpm], X_val_extended[col_gpm], X_test_extended[col_gpm])
    
    return x_train,x_val,x_test

def rename_time(df:pd.DataFrame, version = 1):
    if version == 1:
        df['time'] = df['duration']
        df = df.drop(columns=['duration'])
        return df
    elif version == 2:
        df['time'] = df['mean_latency']
        df = df.drop(columns=['mean_latency'])
        return df
        
from svm import train_bestmodel_svr,scale_log_data_targets,save_svm_prediction
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'SPARQL QPP model on LSQ dataset',
                    description = 'Runs experiments',
                    epilog = '...')
    parser.add_argument('task')
    parser.add_argument('-d','--data_path')
    parser.add_argument('-l','--latency_col', default='duration')
    args = parser.parse_args()
    if args.task == 'nn':
        train, val, test =  process_nn_data(args)
        #train, val, test = rename_time(train), rename_time(val), rename_time(test)
        train, val, test = rename_time(train, version=2), rename_time(val, version=2), rename_time(test, version=2)
        train = train.dropna()
        val = val.dropna()
        test = test.dropna()
        stats, model, scaler = executar(train,val,test,train_aec=True)
        print("AEC + ANN")
        print(stats)
        printSTDVARMEAN(stats, "AEC + ANN")
        save_prediction(train,scaler,model,args.data_path+"nn_train_pred.csv")
        save_prediction(val,scaler,model,args.data_path+"nn_val_pred.csv")
        save_prediction(test,scaler,model,args.data_path+"nn_test_pred.csv")
    #model.predict(train_df.iloc[0])
    elif args.task == 'svm':
        
        train, val,test = process_svm_data(args)
        #train, val,test = svm_pre_process(train,val,test)
        train, val, test = rename_time(train), rename_time(val), rename_time(test)
        #This is not original code
        train = train.dropna()
        val = val.dropna()
        test = test.dropna()
        #dftable, results_baseline= search_hiperparameter_svr(train,val,test)
        result_table, result_baseline_model, scaler, best_model=train_bestmodel_svr(340, 0.10, train, val, test)
        idx = np.argmin(list(result_table.loc[4]))
        print(result_table[idx])
        
        x_train, x_val, x_test , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log = scale_log_data_targets(train, val, test)
        save_svm_prediction(train,x_train,scaler,best_model,args.data_path+ "svm_train_pred.csv")
        save_svm_prediction(val,x_val,scaler,best_model,args.data_path+"svm_val_pred.csv")
        save_svm_prediction(test,x_test,scaler,best_model,args.data_path+"svm_test_pred.csv")
    elif args.task == 'svm-new':
        process_svm_lsq_data(args)
        exit()
    elif args.task == 'two-step':
        train, val,test = process_svm_data(args)
        #train, val,test = svm_pre_process(train,val,test)
        train, val, test = rename_time(train), rename_time(val), rename_time(test)
        cols = get_std_data_cols(train)
        train_times =  np.reshape(train['time'].to_numpy(),(-1,1))
        cluster = XMeansCluster( train_times, 5)
        train = cluster.assign_class(train)
        print(cols)
        t_classifier = TimeClassifer(train[cols], train['class_labels'], 'rbf')
        #print(train['class_labels'].describe())
        
    elif args.task == 'nn-no-ged':
        
        train, val, test =  process_nn_data(args)
        train, val, test = rename_time(train), rename_time(val), rename_time(test)
        train = train.dropna()
        val = val.dropna()
        test = test.dropna()
        
        skip_columns = ['cls_0',
       'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_6', 'cls_7', 'cls_8',
       'cls_9', 'cls_10', 'cls_11', 'cls_12', 'cls_13', 'cls_14', 'cls_15',
       'cls_16', 'cls_17', 'cls_18', 'cls_19', 'cls_20', 'cls_21', 'cls_22',
       'cls_23', 'cls_24']
        train = train.drop(columns=skip_columns)
        val = val.drop(columns=skip_columns)
        test = test.drop(columns=skip_columns)
        
        stats, model, scaler = executar(train,val,test,train_aec=True)
        print("AEC + ANN")
        print(stats)
        printSTDVARMEAN(stats, "AEC + ANN")
        save_prediction(train,scaler,model,args.data_path+"nn_no_ged_train_pred.csv")
        save_prediction(val,scaler,model,args.data_path+"nn_no_ged_val_pred.csv")
        save_prediction(test,scaler,model,args.data_path+"nn_no_ged_test_pred.csv")