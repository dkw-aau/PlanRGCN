import pandas as pd
from sklearn.svm import NuSVR
#from keras import backend as K
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import time
import os
import numpy as np
from keras import backend as K

PATH = '/code/data/11Dec/data/'
query_log_path = '/code/data/datasetlsq_30000.csv'

def load_dataset(path, log_path):
    val_paths = {
        'cl': path+'val_cluster.csv',
        'alg': path+'val_algebra.csv'
    }
    train_paths = {
        'cl': path+'traindata_cluster.csv',
        'alg': path+'train_algebra.csv'
    }
    test_paths = {
        'cl': path+'test_cluster.csv',
        'alg': path+'test_algebra.csv'
    }
    log = query_id_time(log_path)
    train_df = merge(load_df(train_paths['alg'], train_paths['cl']),log)
    val_df = merge(load_df(val_paths['alg'], val_paths['cl']),log)
    test_df = merge(load_df(test_paths['alg'], test_paths['cl']),log)
    lens = [len(train_df), len(val_df), len(test_df)]
    df = pd.concat([train_df,val_df,test_df])

    nunqie = df.apply(pd.Series.nunique)
    cols_to_drop = nunqie[nunqie==1].index
    print("Remove columns with same values: {}".format(cols_to_drop))
    df = df.drop(cols_to_drop, axis=1)
    df = df[df['time'] < 15000]
    X_temp, X_test = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
    X_train, X_val = train_test_split(
        X_temp, test_size=0.30, random_state=42,shuffle=True)
    print("Shapes : Train: {} Val: {}, Test: {}".format(X_train.shape, X_val.shape, X_test.shape))
    return X_train,X_val,X_test

def load_dataset_preserving_split(path, log_path):
    val_paths = {
        'cl': path+'/val_cluster.csv',
        'alg': path+'/val_algebra.csv'
    }
    train_paths = {
        'cl': path+'/traindata_cluster.csv',
        'alg': path+'/train_algebra.csv'
    }
    test_paths = {
        'cl': path+'/test_cluster.csv',
        'alg': path+'/test_algebra.csv'
    }
    log = query_id_time(log_path)
    train_df = merge(load_df(train_paths['alg'], train_paths['cl']),log)
    val_df = merge(load_df(val_paths['alg'], val_paths['cl']),log)
    test_df = merge(load_df(test_paths['alg'], test_paths['cl']),log)

    lens = [len(train_df), len(val_df), len(test_df)]
    df = pd.concat([train_df,val_df,test_df])

    nunqie = df.apply(pd.Series.nunique)
    cols_to_drop = nunqie[nunqie==1].index
    print("Remove columns with same values: {}".format(cols_to_drop))
    df = df.drop(cols_to_drop, axis=1)
    #df = df[df['time'] < 15000] #this is what they do in the works.
    #X_temp, X_test = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
    #X_train, X_val = train_test_split(
    #    X_temp, test_size=0.30, random_state=42,shuffle=True)
    X_train = df[0:lens[0]]
    X_val = df[lens[0]:lens[0]+lens[1]]
    X_test = df[lens[1]+lens[0]:lens[1]+lens[0]+lens[2]]
    X_train = X_train[X_train['time'] < 15000]
    X_val = X_val[X_val['time'] < 15000]
    X_test = X_test[X_test['time'] < 15000]

    assert len(df) == (len(X_train)+len(X_val)+len(X_test))
    print("Shapes : Train: {} Val: {}, Test: {}".format(X_train.shape, X_val.shape, X_test.shape))
    return X_train,X_val,X_test

def load_cluster_file(fp):
    data = {}
    with open(fp, 'r') as f:
        columns = f.readline().replace('\n','').split(',')
        data[columns[0]] = []
        for l_i,line in enumerate(f.readlines()):
            line = line.replace('\n','').replace('[','').replace(']','')
            spl = line.split(',',1)
            data[columns[0]].append(spl[0])
            spl = spl[1].split(',')
            for i, x in enumerate(spl):
                if l_i == 0:
                    data['cls_{}'.format(i)] = []
                data['cls_{}'.format(i)].append((1/(1+float(x))))
            #for value,col_name in zip(spl,columns):
            #    data[col_name].append(value)
    df = pd.DataFrame.from_dict(data)
    return df

def string_to_vec(vec):
    new = []
    for n in vec.split(','):
        new.append(float(n))
    return new
def load_algebra(fp):
    df = pd.read_csv(fp)
    return df

def merge(df_alg,df_cls):
    return df_cls.merge(df_alg,on='id')

def load_df(path_alg,path_clus):
    df_clus = load_cluster_file(path_clus)
    df_alg = load_algebra(path_alg)

    if 'query_id' in df_alg.columns:
        df_alg = df_alg.rename(columns={'query_id':'id'})
    return merge(df_alg,df_clus)

def query_id_time(query_log_path):
    query_log = pd.read_csv(query_log_path)
    query_log = query_log.loc[:,['s','runTimeMs']]
    return query_log.rename(columns={'s':'id','runTimeMs':'time'})



def split_data(df):

    gpf_algebra = df.drop(columns=["id"])

    nunique = gpf_algebra.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    print("Remove columns with same values: {}".format(cols_to_drop))
    gpf_algebra = gpf_algebra.drop(cols_to_drop, axis=1)
    #/////
    #new_cols = list(gpf_algebra.columns[:20]) + ['cls_'+str(i) for i in list(range(0,25))]
    #gpf_algebra.columns = new_cols
    #/////
    #gpf_algebra = gpf_algebra[gpf_algebra['time'] < 15000]
    X_temp, X_test = train_test_split(gpf_algebra, test_size=0.25, random_state=42, shuffle=True)
    #/////
    X_train, X_val = train_test_split(
    X_temp, test_size=0.30, random_state=42,shuffle=True)
    print("Shapes : Train: {} Val: {}, Test: {}".format(X_train.shape, X_val.shape, X_test.shape))

def coeff_determination(y_true, y_pred):
    """Coefficient of determination to use in metrics calculated by epochs"""
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def rmse(y_true, y_pred):
    """RMSE to use in metrics calculated by epochs"""
    return K.exp(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))

def coeff_determination_simple(y_true, y_pred):
    SS_res =  np.sum(np.square( y_true - y_pred ))
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def normalize_target(y_train_log, y_val_log, y_test_log):
    """
    Normalize data using StandardScaler.

    return scaler object; values of train,val and test sets standarized.
    """
    #Standarización del target
    scaler = StandardScaler()
    y_train_log_std = scaler.fit_transform(y_train_log)
    y_val_log_std = scaler.transform(y_val_log)
    y_test_log_std = scaler.transform(y_test_log)
    return scaler, y_train_log_std, y_val_log_std, y_test_log_std

def get_metrics(model, scaler, x_data, y_true_data, label_set="Data"):
    y_pred = np.exp(scaler.inverse_transform(model.predict(x_data).reshape(-1, 1)))
    rmse = np.sqrt(mean_squared_error(y_true_data, y_pred))
    r2 = r2_score(y_true_data, y_pred)
    print("RMSE "+label_set, rmse)
    print("R2 SCORE "+label_set, r2)
    return rmse, r2


from util import normalizaAlgebra
def baseline_svr(C, nu, Xdata, Ydata):

    sv = NuSVR(C=C, nu=nu)

    t0=time()
    print("before train: Init time: {}".format(round(t0,3)))

    sv.fit(Xdata, Ydata)
    t1=time()

    print("after train, finish time: {}".format(round(t1,3)))
    print("training time {}",format(round(t1-t0,3)))
    return [sv, round(t1-t0, 3)]

def get_metrics_svr_model(sv, scalery, x_train, y_train, y_log_std):

    y_train_hat_svr = scalery.inverse_transform(sv.predict(x_train).reshape(-1, 1))

    #MSE for valid
    mse_svr_curr      = np.sqrt(mean_squared_error(y_train, y_train_hat_svr))

    scores_train_curr = sv.score(x_train, y_log_std)

    return mse_svr_curr, scores_train_curr



def scale_log_data_targets(df_train, df_val, df_test):

    y_train = df_train['time'].values.reshape(-1, 1)
    y_val   = df_val['time'].values.reshape(-1, 1)
    y_test  = df_test['time'].values.reshape(-1, 1)

    y_val_log = np.log(y_val)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    y_train_log_min = np.min(y_train_log)
    y_train_min = np.min(y_train)

    y_train_log_max = np.max(y_train_log)
    y_train_max = np.max(y_train)

    print("targets min:{} max: {}".format(y_train_min, y_train_max))
    print("targets in log scale min:{} max: {}".format(y_train_log_min, y_train_log_max))

    #return df_train.drop(columns=['time','id']), df_val.drop(columns=['time','id']), df_test.drop(columns=['time','id']) , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log
    """drop_column = []
    for col in df_train.columns:
        print(col)
        if col in ['time','execTime','id','queryID']:
            drop_column.append(col)"""
    #'joins_count', 'filter_uri', 'filter_num', 'filter_literal', are filtered because they should be additional features.
    retain = ['triple', 'bgp', 'leftjoin', 'union', 'filter', 'graph', 'extend',
                         'minus', 'order', 'project', 'distinct',  'group', 'slice',
                         'treesize','cls_0', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'cls_6', 'cls_7', 'cls_8', 'cls_9', 'cls_10', 'cls_11', 'cls_12',
        'cls_13', 'cls_14', 'cls_15', 'cls_16', 'cls_17', 'cls_18', 'cls_19', 'cls_20', 'cls_21', 'cls_22', 'cls_23', 'cls_24']
    return df_train[retain], df_val[retain], df_test[retain] , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log
    #return df_train.drop(columns=drop_column), df_val.drop(columns=drop_column), df_test.drop(columns=drop_column) , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log

def train_bestmodel_svr(C, nu, df_train, data_val, data_test):
    import random
    result_baseline_model = []
    min_rmse = 100000000000000
    dftable = pd.DataFrame(columns=[])
    x_train, x_val, x_test , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log = scale_log_data_targets(df_train, data_val, data_test)
    print("Shape datasets x: {}".format(x_train.shape))
    print("Shape datasets xval: {}".format(x_val.shape))
    print("Shape datasets xtest: {}".format(x_test.shape))
    print("Columns datasets after normalize.", x_train.columns)
    # scale target using StandarSacaler
    scalery, y_train_log_std, y_val_log_std, y_test_log_std = normalize_target(y_train_log, y_val_log, y_test_log)
    #scalery, y_train_log_std, y_val_log_std, y_test_log_std = normalize_target(y_train, y_val, y_test)

    for i in range(1, 10):

        #result_baseline_model
        #Train model
        sv, training_time = baseline_svr(C,nu, x_train.values, y_train_log_std)

        rmse_train, r2_train = get_metrics_svr_model(sv, scalery, x_train.values, y_train, y_train_log_std)
        rmse_val, r2_val = get_metrics_svr_model(sv, scalery, x_val.values, y_val, y_val_log_std)
        rmse_test, r2_test = get_metrics_svr_model(sv, scalery, x_test.values, y_test, y_test_log_std)

        print("RMSE train: {}, R2 train {}".format(rmse_train, r2_train))
        print("RMSE val: {}, R2 val {}".format(rmse_val, r2_val))
        print("RMSE test: {}, R2 test:{}".format(rmse_test, r2_test))
        dftable = dftable.append(pd.Series([
            C, nu,
            rmse_train, rmse_val, rmse_test,
            r2_train, r2_val, r2_test,
            training_time]), ignore_index=True
        )
        if rmse_test < min_rmse:
            best_model = sv
            min_rmse = rmse_test
        result_baseline_model.append(sv)
    return dftable, result_baseline_model,scalery, best_model

def search_hiperparameter_svr(df_train, data_val, data_test):
    import random
    result_baseline_model = []
    dftable = pd.DataFrame(columns=[])
    x_train, x_val, x_test , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log = scale_log_data_targets(df_train, data_val, data_test)
    print("Shape datasets x: {}".format(x_train.shape))
    print("Shape datasets xval: {}".format(x_val.shape))
    print("Shape datasets xtest: {}".format(x_test.shape))
    print("Columns datasets after normalize.", x_train.columns)
    # scale target using StandarSacaler
    scalery, y_train_log_std, y_val_log_std, y_test_log_std = normalize_target(y_train_log, y_val_log, y_test_log)

    for i in range(1, 10):
        C = random.randrange(100, 350, 20)
        nu = random.randrange(10, 50, 5)/100
        #print(result_baseline_model)
        #Train model
        sv, training_time = baseline_svr(C, nu, x_train.values, y_train_log_std)

        rmse_train, r2_train = get_metrics_svr_model(sv, scalery, x_train.values, y_train, y_train_log_std)
        rmse_val, r2_val = get_metrics_svr_model(sv, scalery, x_val.values, y_val, y_val_log_std)
        #     rmse_test, r2_test = get_metrics_svr_model(sv, scalery, x_test, y_test, y_test_hat_svr)

        print("RMSE train: {}, R2 train {}".format(rmse_train, r2_train))
        print("RMSE val: {}, R2 val {}".format(rmse_val, r2_val))
        #     print("MSE test: {}, R2 test:{}".format(mse_svr_test_curr, scores_test_curr))
        """dftable = pd.concat([dftable,pd.Series([
            C, nu, rmse_train, rmse_val, 0,
            r2_train, r2_val, 0,
            training_time])])"""
        dftable = dftable.append(pd.Series([
            C, nu, rmse_train, rmse_val, 0,
            r2_train, r2_val, 0,
            training_time]), ignore_index=True
        )
        result_baseline_model.append(sv)
    return dftable, result_baseline_model

def save_svm_prediction(df,xs,scaler,model,path_to_save):
    y_train_hat_svr = scaler.inverse_transform(np.exp( model.predict(xs).reshape(-1, 1)))
    #y_train_hat_svr = scaler.inverse_transform( model.predict(xs).reshape(-1, 1))
    
    df = pd.DataFrame({'id':df['id'],'time':df['time'],'svm_prediction':y_train_hat_svr.flatten()})
    df.to_csv(path_to_save)


if __name__ == "__main__":
    #df_cls = load_cluster_file('val_cluster.csv')
    #df_alg = load_algebra('/Users/abirammohanaraj/Documents/GitHub/qpp/data/experiment2/val_queries_algebra.csv')
    #df = merge(df_alg,df_cls)

    #Untested environment variable modifications:
    PATH = os.environ['datapath']+'/'+os.environ['experiment_name']+'/data/'
    query_log_path = os.environ['datapath']+'/datasetlsq_30000.csv'
    train_df,val_df,test_df = load_dataset_preserving_split(PATH, query_log_path)

    #result_table,model_=search_hiperparameter_svr(train_df,val_df,test_df)
    result_table, result_baseline_model, scaler, best_model= train_bestmodel_svr(260,0.4,train_df,val_df,test_df)
    idx = np.argmin(list(result_table.loc[4]))
    print(result_table[idx])
    #print(df.head(5))


    x_train, x_val, x_test , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log = scale_log_data_targets(train_df, val_df, test_df)
    save_svm_prediction(train_df,x_train,scaler,best_model,os.environ['datapath'] +'/'+ os.environ['experiment_name']+"/results/svm_train_pred.csv")
    save_svm_prediction(val_df,x_val,scaler,best_model,os.environ['datapath'] +'/'+ os.environ['experiment_name']+"/results/svm_val_pred.csv")
    save_svm_prediction(test_df,x_test,scaler,best_model,os.environ['datapath'] +'/'+ os.environ['experiment_name']+"/results/svm_test_pred.csv")


    #pd.DataFrame({'prediction':y_pred.tolist(),'actual':x_train['execTime'].tolist()}).to_csv('/code/data/11Dec/data/svm/normal_svm_train_preds.csv',index=False)

    #y_pred = np.exp(scaler.inverse_transform(best_model.predict(x_val).reshape(-1, 1)))
    #y_pred = y_pred.flatten()
    #pd.DataFrame({'prediction':y_pred.tolist(),'actual':x_val['execTime'].tolist()}).to_csv('/code/data/11Dec/data/svm/normal_svm_val_preds.csv',index=False)

    #y_pred = np.exp(scaler.inverse_transform(best_model.predict(x_test).reshape(-1, 1)))
    #y_pred = y_pred.flatten()
    #pd.DataFrame({'prediction':y_pred.tolist(),'actual':x_test['execTime'].tolist()}).to_csv('/code/data/11Dec/data/svm/normal_svm_test_preds.csv',index=False)
