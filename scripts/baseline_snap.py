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


import numpy as np
import matplotlib.pyplot as plt

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
#######################

parse = argparse.ArgumentParser(prog="RegressionSnapper", description="Snaps regression results to classes")
parse.add_argument('-s','--split_dir', help='Folder name in path to where the test_sampled.tsv file is located')
parse.add_argument('-t','--time_intervals', default=None,type=int, help='the amount of time intervals. Choices are 3, 5!')
parse.add_argument('-f','--pred', help='The prediction files')
parse.add_argument('-o', '--outputfolder', default=None, help='the path where the result folder should be created')
parse.add_argument('--set', default='test_sampled.tsv', help='the path where the result folder should be created')
parse.add_argument('--objective', default=None, help=' the objective.py function')


args = parse.parse_args()
print(args)

if args.objective is not None:
    exec(open(args.objective).read(), globals())

    t_func = lambda x: np.argmax(cls_func(x))
    ResultProcessor.gt_labels = [x for x in range(n_classes)]


if args.outputfolder == None:
    raise Exception("outputfolder must be specified!")
else:
    output_fold = args.outputfolder
    os.makedirs(output_fold, exist_ok=True)

if args.objective is None:
	name_dict, temp_c_func = time_ints(args.time_intervals)
	cls_func = lambda x: np.argmax(temp_c_func(x))

path = args.split_dir
pred_path = args.pred
split_path = f"{args.split_dir}/{args.set}"

df = pd.read_csv(pred_path)
df['id'] = df['id'].apply(lambda x: x[20:])
print(df)
df['time_cls'] = df['time'].apply(t_func)
is_NN = False
is_SVM = False
for x in df.columns:
    if 'nn_pred' in x:
        is_NN=True
    elif 'svm' in x:
        is_SVM = True

if is_NN:
    o_file = 'nn_pred.csv'
    df['planrgcn_prediction'] = df['nn_prediction'].apply(t_func)
else:
    o_file = 'svm_pred.csv'
    df['planrgcn_prediction'] = df['svm_prediction'].apply(t_func)

df = df[['id','time_cls','planrgcn_prediction']]
print(df)
df.to_csv(os.path.join(output_fold, o_file), index=False)
