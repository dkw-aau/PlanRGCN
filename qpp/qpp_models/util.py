import math

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np

def filter_only_string_non_empty(x):
    if(type(x) == float):
        return False
    return x.replace("EMPTY_VALUE","")

def uri_2_index_seq(x,uri2Index):
    """transform uris to sequences"""
    lista  = [uri2Index[a] for a in x]
    #     print(lista)
    return lista
def get_joins(x):
    if type(x) != str:
        return []
    lista = [a for a in x.split("B?B") if a != ""]
    if len(lista) == 1:
        return []
    valsjoins = []
    for i in range(len(lista))[::2]:
        valsjoins.append([lista[i],lista[i+1]])
    return valsjoins

def convert_string2dict(x):
    """Convertir string en un dict donde las keys son los pareces y los valores los impares, ignora llaves sin valores y llaves empty"""
    if type(x) is float and math.isnan(x):
        return {}
    if x is None:
        return {}
    values = x.split("B?B")
    rango = range(len(values))
    resp = {}
    for i in rango:
        if(i%2 == 0 and values[i] != ""):
            try:
                resp[values[i]] = values[i+1]
            except:
                print("error con ", values[i], x)
    return resp

def get_sum_values(x):
    total = 0
    for val in list(x.values()):
        total += float(val)
    return total

def selectivity(val,low, high, distinct, operator):
    """
    Calculate Select Estimation  sel_A<=c """
    if(operator == "<="):
        select = (val-low)/(high - low)
    elif(operator == ">="):
        select = 1- ((val-low)/(high - low))
    elif(operator == "="):
        select = (1/distinct)
    else:
        select = (distinct-1)/(distinct)
    return 1
def get_hist_value(data_tpf_histdata,predicate, operator, on, value):
    """
    predicate: predicado uri para extraer la información del histograma.
    operator:
    value: valor para filtrar, si es
    return el valor.
    """
    if(value == "ALL"):
        #Si es ALL la selectividad es 1
        return 1

    data = data_tpf_histdata[data_tpf_histdata['predicate'] == predicate]
    data = data[data['on'] == on]
    if data.shape[0] > 0:
        hist_data = data['hist_array'].values[0]
        distinct = len(hist_data)
        if distinct == 0:
            # Todo, revisar que pasa en caso de que distinct es cero
            return 0
        type_row = data['type'].values[0]
        if type_row == "uri":
            #Todo ver que se hace con la selectividad, aqui, si es el total de duplicados sobre el total de elementos o total
            if value in hist_data:
                #return float(hist_data[value])/distinct
                return float(1)/distinct
            else:
                # Suponemos que el valor es 1 si no se muestreo en el hist.
                return float(1)/distinct
        elif type_row == "numeric":
            min_v = float(hist_data['min'])
            max_v = float(hist_data['max'])
            distinct_v = float(hist_data['distinct'])
            print(hist_data)
            try:
                value = float(value)
            except ValueError:
                print("[{}] input is not a number. It's a string".format(value))
                #Todo verificar que hacer cuando el supuesto número es un str no numerico, de momento devolvemos  1/distinct
                return float(1)/distinct_v
            return selectivity(value, min_v, max_v, distinct_v, operator)
    #      If not return max selectivity
    return 1
def get_pred_list(data):
    pass
def pred_2_hist(x):
    resp = {'uri':0,'num':0,'literal':0}
    if(type(x) != str):
        return resp
    data = [el for el in x.split("??44??") if el != ""]
    #     print(data)
    #     get_hist_value(predicate, operator, value)

    for i in range(len(data)):
        if i%4 == 0:
            uri = data[i]
            operator = data[i+1]
            on = data[i+2]
            val = data[i+3]
            #             select = get_hist_value(uri, operator, on, val)
            if(val.startswith("http")):
                resp['uri'] += 1
            elif(val.isnumeric()):
                resp['num'] += 1
            elif(val != "ALL"):
                resp['literal'] += 1
   
    return resp

def get_filter_by_type(x, typeOf):
    try:
        return x[typeOf]
    except: print(x)


def get_std_data_cols(train_df):
    x_columns_to_norm = ['triple', 'bgp', 'leftjoin', 'union', 'filter', 'graph', 'extend',
                         'minus', 'order', 'project', 'distinct',  'group', 'slice',
                         'treesize'] #extra data , 'filter_uri', 'filter_num', 'filter_literal','joins_count'
    for x in train_df.columns:
        if 'cls' in x:
            x_columns_to_norm.append(x)
    
    cols = x_columns_to_norm
    print(cols)
    return cols
    #Standarizar
def normalizaAlgebra(X_train_extended, X_val_extended, X_test_extended, returnScaler=False):
    #no reduced in my data
    x_columns_to_norm = ['triple', 'bgp', 'leftjoin', 'union', 'filter', 'graph', 'extend',
                         'minus', 'order', 'project', 'distinct',  'group', 'slice',
                         'treesize','joins_count', 'filter_uri', 'filter_num', 'filter_literal'] #'reduced'
    scalerx = StandardScaler()
    x_train_scaled = scalerx.fit_transform(X_train_extended[x_columns_to_norm])
    x_val_scaled = scalerx.transform(X_val_extended[x_columns_to_norm])
    #x_test_scaled = scalerx.fit_transform(X_test_extended[x_columns_to_norm]) //dobble check whether they do this in source, because it's wrong.
    x_test_scaled = scalerx.transform(X_test_extended[x_columns_to_norm])

    scaled_df_train = pd.DataFrame(x_train_scaled, index=X_train_extended.index, columns=x_columns_to_norm)
    scaled_df_val = pd.DataFrame(x_val_scaled, index=X_val_extended.index, columns=x_columns_to_norm)
    scaled_df_test = pd.DataFrame(x_test_scaled, index=X_test_extended.index, columns=x_columns_to_norm)
    if returnScaler:
        return scaled_df_train, scaled_df_val, scaled_df_test, scalerx
    return scaled_df_train, scaled_df_val, scaled_df_test

def joinAlgebraGPM(Train_alg, Val_alg, Test_alg, Train_gpm, Val_gpm, Test_gpm):
    mergedTrain = Train_alg.merge(Train_gpm, left_index=True, right_index=True)
    mergedVal   = Val_alg.merge(  Val_gpm,   left_index=True, right_index=True)
    mergedTest  = Test_alg.merge( Test_gpm,  left_index=True, right_index=True)
    return mergedTrain, mergedVal, mergedTest
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

    return df_train.drop(columns=['time']), df_val.drop(columns=['time']), df_test.drop(columns=['time']) , y_train, y_val, y_test, y_train_log, y_val_log, y_test_log

def printSTDVARMEAN(stastNoAEC, title):
    print(title)
    print("STD")
    print(stastNoAEC.std())
    print("VAR")
    print(stastNoAEC.std()**2)
    print("MEAN")
    print(stastNoAEC.mean())

def re(actual, predicted):
    return np.abs(predicted - actual)/actual

def mre(actual, predicted):
    temp = 0
    for a,p in zip(actual,predicted):
        temp += (np.abs(p - a)/a)
    return temp/len(actual)

def rae(actual,predicted):
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator

def rmse(y_true_data, y_pred):
    return np.sqrt(mean_squared_error(y_true_data, y_pred))