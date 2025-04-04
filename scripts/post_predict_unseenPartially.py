import json
import pandas as pd
import os
import json5
from graph_construction.qp.visitor.DefaultVisitor import DefaultVisitor
from graph_construction.qp.visitor.AbstractVisitor import dispatch
from inductive_query.utils import *
from inductive_query.result_processor import *
import pathlib
from inductive_query.res_proc_helper import *
import importlib
import inductive_query.utils as ih
importlib.reload(ih)
CompletelyUnseenQueryExtractor = ih.CompletelyUnseenQueryExtractor

import inductive_query.res_proc_helper as help
importlib.reload(help)
import inductive_query.result_processor as res_proc
importlib.reload(res_proc)
ResultProcessor = res_proc.ResultProcessor
from inductive_query.res_proc_helper import *
import importlib
import inductive_query.res_proc_helper as indH
importlib.reload(indH)
get_result_processor = indH.get_result_processor
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set_theme(font='serif')
import argparse
import json
import pandas as pd
import os
import json5
from graph_construction.qp.visitor.DefaultVisitor import DefaultVisitor
from graph_construction.qp.visitor.AbstractVisitor import dispatch
from inductive_query.utils import *
from inductive_query.result_processor import *
import pathlib
from inductive_query.res_proc_helper import *
import importlib
import inductive_query.utils as ih
importlib.reload(ih)
CompletelyUnseenQueryExtractor = ih.CompletelyUnseenQueryExtractor

import inductive_query.res_proc_helper as help
importlib.reload(help)
import inductive_query.result_processor as res_proc
importlib.reload(res_proc)
ResultProcessor = res_proc.ResultProcessor
from inductive_query.res_proc_helper import *
import importlib
import inductive_query.res_proc_helper as indH
importlib.reload(indH)
get_result_processor = indH.get_result_processor
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set_theme(font='serif')

from inductive_query.res_proc_helper import get_unseen_result_processor
from graph_construction.query_graph import snap_lat2onehotv2
cls_func = lambda x: np.argmax(snap_lat2onehotv2(x))


from inductive_query.result_processor import ResultProcessor
import inductive_query.pp_qextr as pp_qextr
importlib.reload(pp_qextr)
PPQueryExtractor = pp_qextr.PPQueryExtractor

def get_PP_result_processor(dataset_path, pred_path, split_path, name_dict, approach_name,apply_cls_func=None, split_type='test'):
    ext = PPQueryExtractor(dataset_path)
    #ext.set_test_pp()
    match split_type:
        case 'test':
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_test_PP_files()]
        case 'train':
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_train_PP_files()]
        case 'val':
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_val_PP_files()]
    p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
    p.retain_path(split_path)
    p.retain_ids(unseen_pred_queryID)
    print("PP predicate")
    print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
    print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
    return p
def merge_mean_latency(resultProcessor, path):
    df = pd.read_csv(path, sep='\t')
    df['id'] = df['id'].apply(lambda x: x[20:])
    return pd.merge(resultProcessor.df, df[['id', 'mean_latency']], on='id', how='left')

import numpy as np
import matplotlib.pyplot as plt
def plot_cdf(df, col='mean_latency'):
    values = df[col]
    cumulative= np.linspace(0,1,len(values))
    sorted_data = np.sort(values)
    cumulative_data = np.cumsum(sorted_data) / np.sum(sorted_data)
    plt.plot(sorted_data, cumulative_data)
    plt.xlabel("Cumulative "+col)
    plt.ylabel("Cumulative Proportion")
    plt.title("Cumulative Distribution Function (CDF) of "+col)
    plt.show()
    print(sorted_data)

import importlib
import inductive_query.res_proc_helper as help
importlib.reload(help)
import inductive_query.result_processor as res_proc
importlib.reload(res_proc)
import inductive_query.utils as ih
importlib.reload(ih)
CompletelyUnseenQueryExtractor = ih.CompletelyUnseenQueryExtractor
ResultProcessor = res_proc.ResultProcessor
get_completely_unseen_r_processor = help.get_completely_unseen_r_processor

from graph_construction.query_graph import snap_lat2onehotv2, snap5cls
import pathlib
import importlib
from inductive_query.result_processor import ResultProcessor
import inductive_query.pp_qextr as pp_qextr
importlib.reload(pp_qextr)
PPQueryExtractor = pp_qextr.PPQueryExtractor
def get_PP_result_processor(dataset_path, pred_path, split_path, name_dict, approach_name,apply_cls_func=None, split_type='test'):
    ext = PPQueryExtractor(dataset_path)
    #ext.set_test_pp()
    match split_type:
        case 'test':
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_test_PP_files()]
        case 'train':
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_train_PP_files()]
        case 'val':
            unseen_pred_queryID =[pathlib.Path(x).name for x in ext.get_val_PP_files()]
    p = ResultProcessor(pred_path, approach_name=approach_name,apply_cls_func=apply_cls_func)
    p.retain_path(split_path, remove_prefix=20)
    p.retain_ids(unseen_pred_queryID)
    print("PP predicate")
    print(p.confusion_matrix_to_latex_row_wise(name_dict=name_dict))
    print(p.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
    return p
def clean_latex_tables(c):
    c = c.replace('toprule', 'hline')
    c = c.replace('midrule', 'hline')
    c = c.replace('bottomrule', 'hline')
    
    c = c.replace('Predicted', ' ')
    c = c.replace('Actual', ' ')
    return c

UnseenResultProcessor = help.UnseenResultProcessor
#######################

parse = argparse.ArgumentParser(prog="PartiallyUnseenPredictionProcessor", description="Post Processing of results for data. Formatted from the Notebooks")
parse.add_argument('-s','--split_dir', help='Folder name in path to where the test_sampled.tsv file is located')
parse.add_argument('-t','--time_intervals', default=None,type=int, help='the amount of time intervals. Choices are 3, 5!')
parse.add_argument('-f','--pred', help='The prediction files')
parse.add_argument('-a', '--approach', help='the name of the results')
parse.add_argument('-o', '--outputfolder', default=None, help='the path where the result folder should be created')
parse.add_argument('--set', default='test_sampled.tsv', help='the path where the result folder should be created')
parse.add_argument('--objective', default=None, help=' the objective.py function')


args = parse.parse_args()


if args.objective is not None:
    exec(open(args.objective).read(), globals())
    cls_func = lambda x: np.argmax(temp_c_func(x))
    ResultProcessor.gt_labels = [x for x in range(n_classes)]

def time_ints(t):
    match t:
        case 3:
            ResultProcessor.gt_labels = [0,1,2]
            return {
                0: '0s to 1s',
                1: '1s to 10s',
                2: '$>$ 10s',
            }, snap_lat2onehotv2
        case 5:
            ResultProcessor.gt_labels = [0,1,2,3,4]
            return {
                0: '(0s; 0.004]',
                1: '(0.004s; 1]',
                2: '(1s; 10]',
                3: '(10; timeout]',
                4: 'timeout',
            }, snap5cls

if args.outputfolder == None:
    output_fold = pathlib.Path(args.pred).parent
    output_fold = os.path.join(output_fold, "results")
    os.makedirs(output_fold, exist_ok=True)
else:
    output_fold = args.outputfolder

if args.objective is None:
	name_dict, temp_c_func = time_ints(args.time_intervals)
	cls_func = lambda x: np.argmax(temp_c_func(x))

approach_name = f"{args.approach}"
path = args.split_dir
pred_path = args.pred
split_path = f"{args.split_dir}/{args.set}"

analyser= UnseenResultProcessor(path, pred_path, split_path, name_dict, approach_name, remove_prefix=20)
for t in ['All', 'pred', 'entity']:
    match t:
        case 'All':
            p = analyser.get_partially_unseen_all()
        case "pred":
            p = analyser.get_partially_unseen_predicate()
        case "entity":
            p = analyser.get_partially_unseen_entity()
        
    with open(os.path.join(output_fold,f'{t}_Partial_unseen_conf_matrix_row_wise.txt'),'w') as f:
        c,t = p.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=True, add_sums=True)
        c = clean_latex_tables(c)
        f.write(c)
        f.write('\n')
        f.write(str(t))
exit()

exit()
if 'test' in args.set:
    DBpedia_PP = get_PP_result_processor(path, pred_path, split_path, name_dict, approach_name, split_type='test')
elif 'train' in args.set:
    DBpedia_PP = get_PP_result_processor(path, pred_path, split_path, name_dict, approach_name, split_type='train')
elif 'val' in args.set:
    DBpedia_PP = get_PP_result_processor(path, pred_path, split_path, name_dict, approach_name, split_type='val')
else:
    raise Exception('Incorrect set '+ arg.set)
with open(os.path.join(output_fold,'confusion_matrix_PP_row_wise.txt'),'w') as f:
    c,t = DBpedia_PP.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=True, add_sums=True)
    c = clean_latex_tables(c)
    f.write(c)
    f.write('\n')
    f.write(str(t))
DBpedia_PP.confusion_matrix_to_latex_row_wise(name_dict=name_dict, return_sums=False, add_sums=False,to_latex =False).to_csv(os.path.join(output_fold,'confusion_matrix_PP_row_wise.csv'))


with open(os.path.join(output_fold,'confusion_matrix_PP.txt'),'w') as f:    
    f.write(DBpedia_PP.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict))
DBpedia_PP.confusion_matrix_to_latex(row_percentage=False,name_dict=name_dict,to_latex =False).to_csv(os.path.join(output_fold,'confusion_matrix_PP.csv'))
