from sklearn.svm import NuSVR 
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from svm import load_cluster_file, search_hiperparameter_svr
import pandas as pd
from sklearn.metrics import mean_squared_error

def train_svm(train : pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, prediction_dir = None, seed = 42):
    np.random.seed(seed)
    
    #svm = NuSVR(nu=0.5,C=1, kernel='rbf')
    skip_columns = ['duration','latency','queryID','timestamp','queryString','projectVariables','tripleCount', 'joinVertexCount', 'resultSize','id']
    t = []
    for c in skip_columns:
        if c in train.columns:
            t.append(c)
    skip_columns = t
    del t    
    train_y, train_x= train['duration'],train.drop(columns=skip_columns).to_numpy()
    val_y, val_x = val['duration'], val.drop(columns=skip_columns).to_numpy()
    test_y, test_x = test['duration'], test.drop(columns=skip_columns).to_numpy()
    print(f"Train shape: {train_x.shape}")
    
    steps= [('scaler', StandardScaler()), ('SVM', NuSVR(kernel='rbf'))]
    pipeline = Pipeline(steps)
    parameteres = {'SVM__C':[0.1,10,100], 'SVM__gamma':[0.1]}
    #parameteres = {'SVM__C':[0.001,0.1,10], 'SVM__gamma':[0.1,0.01]}
    grid = GridSearchCV(pipeline, param_grid=parameteres, verbose=4)
    grid.fit(train_x,train_y)
    #pipeline.fit(train_x,train_y)
    
    print( "score = %3.2f" %(grid.score(train_x,train_y)))
    #print(grid.best_params_)
    print( "score = %3.2f" %(grid.score(val_x,val_y)))
    train_rms = mean_squared_error(train_y, grid.predict(train_x), squared=False)
    val_rms = mean_squared_error(val_y, grid.predict(val_x), squared=False)
    test_rms = mean_squared_error(test_y, grid.predict(test_x), squared=False)
    print(f"RMSEs:\n\tTrain: {train_rms}\n\tVal: {val_rms}\n\tTest: {test_rms}")
    if not prediction_dir== None:
        train['svm_prediction'] = grid.predict(train_x)
        train.to_csv(f"{prediction_dir}/train_svm_pred.csv", sep='\t',index=False)
        val['svm_prediction'] = grid.predict(val_x)
        val.to_csv(f"{prediction_dir}/val_svm_pred.csv", sep='\t',index=False)
        test['svm_prediction'] = grid.predict(test_x)
        test.to_csv(f"{prediction_dir}/test_svm_pred.csv", sep='\t',index=False)
    
    #regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.5, kernel='rbf'))
    #regr.fit(train_x,train_y)
    

def prepare_data(train_graph_path,val_graph_path, test_graph_path, train_alg_path, val_alg_path, test_alg_path):
    train_graph = load_cluster_file(train_graph_path).drop_duplicates('id', keep='first')
    val_graph = load_cluster_file(val_graph_path).drop_duplicates('id', keep='first')
    test_graph = load_cluster_file(test_graph_path).drop_duplicates('id', keep='first')

    train_df = pd.read_csv(train_alg_path,sep='\t').drop_duplicates('queryID', keep='first')
    val_df = pd.read_csv(val_alg_path,sep='\t').drop_duplicates('queryID', keep='first')
    test_df = pd.read_csv(test_alg_path,sep='\t').drop_duplicates('queryID', keep='first')

    train_df = train_df.merge(train_graph, left_on='queryID', right_on='id', how='left').fillna(0)
    val_df = val_df.merge(val_graph, left_on='queryID', right_on='id', how='left').fillna(0)
    test_df = test_df.merge(test_graph, left_on='queryID', right_on='id', how='left').fillna(0)
    return train_df,val_df,test_df

def SVM_experiment_runner(train_graph_path,val_graph_path, test_graph_path, train_alg_path, val_alg_path, test_alg_path, prediction_dir=None):
    train_df,val_df,test_df= prepare_data(train_graph_path,val_graph_path, test_graph_path, train_alg_path, val_alg_path, test_alg_path)
    #print(f"Train colums : {train_df.columns}")
    #print(f"Val colums : {val_df.columns}")
    #print(f"test colums : {test_df.columns}")
    train_svm(train_df,val_df,test_df, prediction_dir=prediction_dir)
if __name__ == "__main__":
    train_graph_path = '/qpp/data/final_data/train_graph.txt'
    val_graph_path = '/qpp/data/final_data/val_graph.txt'
    test_graph_path = '/qpp/data/final_data/test_graph.txt'
    
    train_alg_path = '/qpp/data/final_data/train.tsv'
    val_alg_path = '/qpp/data/final_data/val.tsv'
    test_alg_path = '/qpp/data/final_data/test.tsv'
    
    SVM_experiment_runner(train_graph_path,val_graph_path, test_graph_path, train_alg_path, val_alg_path, test_alg_path, prediction_dir='/qpp/data/final_data')