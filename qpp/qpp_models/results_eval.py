import math

from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from matplotlib import pyplot as plt



df_train = '/work/data/final_data/nn_no_ged_train_pred.csv'
df_test = '/work/data/final_data/nn_no_ged_test_pred.csv'
df_val = '/work/data/final_data/nn_no_ged_val_pred.csv'
"""df_train = '/work/data/final_data/nn_train_pred.csv'
df_test = '/work/data/final_data/nn_test_pred.csv'
df_val = '/work/data/final_data/nn_val_pred.csv'"""
train = pd.read_csv(df_train)
test = pd.read_csv(df_test)
val = pd.read_csv(df_val)


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

def snap_to_closest(val, centers = [0.01,0.1, 1, 10, 100]):
    x = None
    x_i = None
    for i,c in enumerate(centers):
        if( x == None or abs(c-val) < x):
            x = c
            x_i = i
    return x_i
def count_centers(df:pd.DataFrame, predcition_column ='nn_prediction'):
    
    print("True: ",df['time_cls'].value_counts())
    print("Pred: ",df[f'{predcition_column}_cls'].value_counts())

def cls_predictions(df:pd.DataFrame, predcition_column ='nn_prediction'):
    df['time_cls'] = df['time'].apply(snap_to_closest)
    df[f'{predcition_column}_cls'] = df[predcition_column].apply(snap_to_closest)
    return df

def get_metrics(df, predcition_column ='nn_prediction', average=None):
    f1 = f1_score(df['time_cls'].to_numpy(),df[f'{predcition_column}_cls'].to_numpy(), average=average)
    recall = recall_score(df['time_cls'],df[f'{predcition_column}_cls'], average=average)
    precision = precision_score(df['time_cls'],df[f'{predcition_column}_cls'], average=average)
    return f1, recall, precision  

def print_metrics(df, predcition_column ='nn_prediction'):
    f1, recall, precision =get_metrics(df, predcition_column=predcition_column, average='macro')
    print(f'\tF1:{f1}\n\tRecall:{recall}\n\tPrecision:{precision}')

def get_cls_df(df, no):
    return df[(df['time_cls']==no)]

def get_cls_df_for_bar(df, no, predcition_column ='nn_prediction'):
    df['time_cls'] = np.where(df['time_cls'] == no, 1,0)
    df[f'{predcition_column}_cls'] = np.where(df[f'{predcition_column}_cls'] == no, 1,0)
    return df
#train['time_cls'] = train['time'].apply(snap_to_closest)
def bxplot_w_info(data, y_label, showPlot=True, plotname='boxplot.png', legend=""):
    plt.clf()
   
    fig, ax = plt.subplots()
    boxes = [
        {
            'label' : "Male height",
            'whislo': 162.6,    # Bottom whisker position
            'q1'    : 170.2,    # First quartile (25th percentile)
            'med'   : 175.7,    # Median         (50th percentile)
            'q3'    : 180.4,    # Third quartile (75th percentile)
            'whishi': 187.8,    # Top whisker position
            'fliers': []        # Outliers
        }
    
    ]
    plt.yscale('log')
    plt.title(legend)
    ax.bxp(data, showfliers=False)
    ax.set_ylabel(y_label)
    if showPlot:
        plt.show()
    else:
        plt.savefig(plotname)
        plt.close()

def calculate_bxp_info(df:pd.DataFrame, label,metric=precision_score,predcition_column ='nn_prediction'):
    errors = metric(df['time'], df [f'{predcition_column}_cls'])
    info = {
        'label' : label,
            'whislo': np.min(errors),    # Bottom whisker position
            'q1'    :  np.quantile(errors,0.25),    # First quartile (25th percentile)
            'med'   :  np.quantile(errors,0.50),    # Median         (50th percentile)
            'q3'    : np.quantile(errors,0.75),    # Third quartile (75th percentile)
            'whishi': np.max(errors),    # Top whisker position
            'fliers': []        # Outliers
    }
    return info
def calculate_bxp_all_cls(df:pd.DataFrame, metric = re, predcition_column ='nn_prediction'):
    values = df['time_cls'].unique()
    boxes = []
    for i in values:
        boxes.append( calculate_bxp_info(get_cls_df_for_bar(df,i), "cls "+str(i), metric=metric, predcition_column=predcition_column))
    return boxes

def plot_all_boxes(df:pd.DataFrame,y_label, metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/bxplot.png', legend="")  :
    boxes = calculate_bxp_all_cls(df, metric=metric, predcition_column=predcition_column) 
    bxplot_w_info(boxes, y_label, showPlot=False, plotname=plotname, legend=legend)

def calculate_bar_data_cls(df:pd.DataFrame,values, predcition_column ='nn_prediction'):
    
    data = {'precision':[], 'recall': [], 'f1':[]}
    #f1, recall, precision =get_metrics(df, predcition_column=predcition_column)
    #print(f1, recall, precision)
    for i in values:
        f1, recall, precision =get_metrics(get_cls_df_for_bar(df,i), predcition_column=predcition_column, average='binary')
        data['precision'].append(precision), data['recall'].append(recall), data['f1'].append(f1)
    return data

def plot_bar_metrics(df: pd.DataFrame, prediction_column = 'nn_prediction', label='Metrics', plotname='/work/data/final_data/figures/metrics.png'):
    values = df['time_cls'].unique()
    values = values.tolist()
    values.extend(df[f'{prediction_column}_cls'].unique())
    values = np.ndarray(values)
    data = calculate_bar_data_cls(df,values, predcition_column =prediction_column)
    values = [f"cls_{v}" for v in values]
    plt.clf()
    x = np.arange(len(values))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics in (%)')
    ax.set_title(label)
    ax.set_xticks(x + width, values)
    ax.legend(loc='upper left', ncols=len(values))
    ax.set_ylim(0, 1)
    plt.savefig(plotname)
    plt.close()
    pass

train = cls_predictions(train)
val = cls_predictions(val)
test = cls_predictions(test)
#print("Training")
count_centers(train)
count_centers(val)
count_centers(test)
#print_metrics(train)
#print_metrics(val)
#print_metrics(test)
#print(len(get_cls_df(train,2)))
exit()
plot_all_boxes(train,'RE', metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/figures/train_bxplot_no_GED.png', legend="Train data without GED")
plot_all_boxes(test,'RE', metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/figures/test_bxplot_no_GED.png', legend="Test data without GED")
plot_all_boxes(val,'RE', metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/figures/val_bxplot_no_GED.png', legend="Val data without GED")
plot_bar_metrics(train, prediction_column = 'nn_prediction', label='Train No GED Metrics', plotname='/work/data/final_data/figures/train_metrics_no_ged.png')
plot_bar_metrics(val, prediction_column = 'nn_prediction', label='Val No GED Metrics', plotname='/work/data/final_data/figures/val_metrics_no_ged.png')
plot_bar_metrics(test, prediction_column = 'nn_prediction', label='Test No GED Metrics', plotname='/work/data/final_data/figures/test_metrics_no_ged.png')

#bxplot_w_info(None,'test',showPlot=False, plotname='/work/data/final_data/bxplot.png')
#print(train['time_cls'].value_counts())
#print(re(train['time'],train['nn_prediction']).mean())
#print(re(val['time'],val['nn_prediction']).mean())
#print(re(test['time'],test['nn_prediction']).mean())

df_train = '/work/data/final_data/nn_train_pred.csv'
df_test = '/work/data/final_data/nn_test_pred.csv'
df_val = '/work/data/final_data/nn_val_pred.csv'
train = pd.read_csv(df_train)
test = pd.read_csv(df_test)
val = pd.read_csv(df_val)
train = cls_predictions(train)
val = cls_predictions(val)
test = cls_predictions(test)
plot_all_boxes(train,'RE', metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/figures/train_bxplot.png', legend="Train data")
plot_all_boxes(test,'RE', metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/figures/test_bxplot.png', legend="Test data")
plot_all_boxes(val,'RE', metric = re, predcition_column ='nn_prediction',plotname='/work/data/final_data/figures/val_bxplot.png', legend="Val data")
plot_bar_metrics(train, prediction_column = 'nn_prediction', label='Train Metrics', plotname='/work/data/final_data/figures/train_metrics.png')
plot_bar_metrics(val, prediction_column = 'nn_prediction', label='Val Metrics', plotname='/work/data/final_data/figures/val_metrics.png')
plot_bar_metrics(test, prediction_column = 'nn_prediction', label='Test Metrics', plotname='/work/data/final_data/figures/test_metrics.png')