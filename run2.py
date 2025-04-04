"""
Backup of old code in run.py
"""
from load_balance.result_analysis.adcanal import ADMCTRLAnalyser


def evaluate_DBpedia_adm_ctrl_script(print_results=True):
    objective_file='/data/DBpedia_3_class_full/admission_control/planrgcn_44/../../objective.py'

    test_sampled = '/data/DBpedia_3_class_full/test_sampled.tsv'
    a = ADMCTRLAnalyser(test_sampled, objective_file=objective_file)

    # TODO use workload 4 - workload 2 should not contain correct data, and workload 3 baseline did not load test workload correctly.
    base_path = '/data/DBpedia_3_class_full/admission_control/workload4_21'
    base_path = '/data/DBpedia_3_class_full/admission_control/workload6'
    plan_res_adm = f'{base_path}/planrgcn'
    nn_res_adm = f'{base_path}/nn'
    svm_res_adm = f'{base_path}/svm'
    a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'DBpedia')
    a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'DBpedia')
    a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'DBpedia')

    if print_results:
        a.process_results()
    return a
#evaluate_DBpedia_adm_ctrl_script()

def evaluate_wikidata_adm_ctrl_script(print_results=True):
    objective_file='/data/DBpedia_3_class_full/admission_control/planrgcn_44/../../objective.py'

    test_sampled = '/data/DBpedia_3_class_full/test_sampled.tsv'
    a = ADMCTRLAnalyser(test_sampled, objective_file=objective_file)
    dataset= 'Wikidata'
    # TODO use workload 4 - workload 2 should not contain correct data, and workload 3 baseline did not load test workload correctly.
    plan_res_adm = '/data/wikidata_3_class_full/admission_control/workload3_21/planrgcn'
    nn_res_adm = '/data/wikidata_3_class_full/admission_control/workload2_21/nn'
    svm_res_adm = '/data/wikidata_3_class_full/admission_control/workload2_21/svm'
    a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = dataset)
    a.evaluate_dataset(nn_res_adm, 'NN', dataset = dataset)
    a.evaluate_dataset(svm_res_adm, 'SVM', dataset = dataset)

    if print_results:
        a.process_results()
    return a
#evaluate_wikidata_adm_ctrl_script()
#evaluate_DBpedia_adm_ctrl_script()
def workload_tester(workload_file):
    import pickle
    import sklearn
    #code that checks different aspect of the workload
    queries, arrival_times = pickle.load(open(workload_file, 'rb'))


    #Check whether the assigned predictions are correct
    predictions = []
    true_cls = []
    for q in queries:
        predictions.append(q.time_cls)
        true_cls.append(q.true_time_cls)
    confs = sklearn.metrics.confusion_matrix(true_cls, predictions, labels=[0,1,2])

    #Check different metrics that can be worth checking
    slow_preds = []
    for q in queries:
        if q.time_cls == 2:
            slow_preds.append(q.true_time_cls)

    len(slow_preds), len([x for x in slow_preds if x == 0 or x == 1])

#workload_tester('/data/DBpedia_3_class_full/admission_control/workload2_/planrgcn/workload.pck')





exit()








objective_file='/data/DBpedia_3_class_full/admission_control/planrgcn_44/../../objective.py'

test_sampled = '/data/DBpedia_3_class_full/test_sampled.tsv'
a = ADMCTRLAnalyser(test_sampled, objective_file=objective_file)


plan_res_adm = '/data/DBpedia_3_class_full/admission_control/workload2_/planrgcn'
nn_res_adm = '/data/DBpedia_3_class_full/admission_control/workload2_/nn'
##svm_res_adm = '/data/wikidata_3_class_full/admission_control/workload1_21/svm'
a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'DBpedia')
a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'DBpedia')
#a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'Wikidata')

a.process_results()
exit()


plan_res_adm = '/data/wikidata_3_class_full/admission_control/workload1_21/planrgcn'
nn_res_adm = '/data/wikidata_3_class_full/admission_control/workload1_21/nn'
svm_res_adm = '/data/wikidata_3_class_full/admission_control/workload1_21/svm'
a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'Wikidata')
a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'Wikidata')
a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'Wikidata')

plan_res_adm = '/data/DBpedia_3_class_full/admission_control/workload1_/planrgcn'
nn_res_adm = '/data/DBpedia_3_class_full/admission_control/workload1_/nn'
svm_res_adm = '/data/DBpedia_3_class_full/admission_control/workload1_/svm'
a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'DBpedia')
a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'DBpedia')
a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'DBpedia')

a.process_results()








exit()
plan_res_adm = '/data/DBpedia_3_class_full/admission_control/planrgcn_44'
nn_res_adm = '/data/DBpedia_3_class_full/admission_control/nn_44'
svm_res_adm = '/data/DBpedia_3_class_full/admission_control/svm_44'

a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'DBpedia')
a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'DBpedia')
a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'DBpedia')

plan_res_adm = '/data/wikidata_3_class_full/admission_control/planrgcn_44'
nn_res_adm = '/data/wikidata_3_class_full/admission_control/nn_44'
svm_res_adm = '/data/wikidata_3_class_full/admission_control/svm_44'
a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'Wikidata')
a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'Wikidata')
a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'Wikidata')

a.process_results()



exit()
# Results for wikidata full query logs



exit()


import pandas as pd

from PlanRGCN.query_log_splitting.data_splitter import *
from qpp.qpp_new.qpp_new.feature_combiner import create_different_k_ged_dist_matrix
from query_class_analyzer import QueryClassAnalyzer,FineGrainedQueryClassAnalyzer
from trainer.new_post_predict import QPPResultProcessor

# from unseen_splitter import UnseenSplitter
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/PlanRGCN/', '/PlanRGCN/PlanRGCN/feature_extraction', '/PlanRGCN/PlanRGCN/feature_representation', '/PlanRGCN/PlanRGCN/graph_construction', '/PlanRGCN/PlanRGCN/trainer', '/PlanRGCN/PlanRGCN/query_log_splitting', '/PlanRGCN/inductive_query'])
# analyze =QueryClassAnalyzer('/data/DBpediaV2/train_sampled.tsv', '/data/DBpediaV2/val_sampled.tsv','/data/DBpediaV2/test_sampled.tsv','/data/DBpediaV2/data_splitter.pickle')


nn_pred = '/data/DBpedia_3_class_full/baseline/nn/k25/nn_test_pred.csv'
svm_pred = '/data/DBpedia_3_class_full/baseline/svm/test_pred_svm.csv'
plan_pred = '/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'

nn_df = pd.read_csv(nn_pred)
svm_df = pd.read_csv(svm_pred)
plan_df = pd.read_csv(plan_pred)


def plot_predictions_vs_actuals(nn_df, svm_df, plan_df, path = None):
    plan_df['time'] = plan_df['time_cls'].apply(lambda x: 0 if x == 0 else 10 if x == 1 else 900)
    plan_df['pred'] = plan_df['planrgcn_prediction'].apply(lambda x: 0 if x == 0 else 10 if x == 1 else 900)

    # Create a figure and axis
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))

    # Plot predictions vs actuals for each dataframe in the same plot
    plt.scatter(nn_df['time'], nn_df['nn_prediction'], color='blue', alpha=0.5, label='NN')
    plt.scatter(svm_df['time'], svm_df['svm_prediction'], color='green', alpha=0.5, label='SVM')
    plt.scatter(plan_df['time'], plan_df['pred'], color='orange', alpha=0.5, label='PlanRGCN')

    # Plot a perfect prediction line
    all_actuals = pd.concat([nn_df['time'], svm_df['time'], plan_df['time']])
    plt.plot([all_actuals.min(), all_actuals.max()],
             [all_actuals.min(), all_actuals.max()], 'r--', label='Perfect Prediction')

    # Add vertical lines for thresholds
    for threshold in [1,10,900]:
        plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold {threshold}')
        plt.axvline(y=threshold, color='grey', linestyle='--', label=f'Threshold {threshold}')

    plt.xscale('log')
    plt.yscale('log')
    # Set plot labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values for Multiple Models')

    # Add legend
    plt.legend()

    if path is not None:
        plt.savefig(path)
    # Show the plot
    plt.show()

exit()


"""p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py', dataset="DBpedia")
p.evaluate_dataset(,

p.evaluate_dataset(,
p.evaluate_dataset(,
print(p.process_results())



exit()
train_file = '/data/DBpediaV2/train_sampled.tsv'
val_file = '/data/DBpediaV2/val_sampled.tsv'
test_file = '/data/DBpediaV2/test_sampled.tsv'
obj_file = '/data/DBpediaV2/plan01/objective.py'




train_file = '/data/DBpedia_3_class_full/train_sampled.tsv'
val_file = '/data/DBpedia_3_class_full/val_sampled.tsv'
test_file = '/data/DBpedia_3_class_full/test_sampled.tsv'


train_file = '/data/wikidata_3_class_full/train_sampled.tsv'
val_file = '/data/wikidata_3_class_full/val_sampled.tsv'
test_file = '/data/wikidata_3_class_full/test_sampled.tsv'
obj_file = '/data/DBpedia_3_class_full/objective.py'
#FineGrainedQueryClassAnalyzer(train_file, val_file, test_file, data_split_path=None, objective_file=obj_file)



exit()
p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py', dataset="DBpedia")
p.evaluate_dataset(,

p.evaluate_dataset(,
p.evaluate_dataset(,
print(p.process_results())



# UnseenSplitter('/data/DBpediaV2/train_sampled.tsv', '/data/DBpediaV2/val_sampled.tsv', '/data/DBpediaV2/test_sampled.tsv','/data/DBpediaV2/data_splitter.pickle','/data/DBpediaV2/n_train_sampled.tsv', '/data/DBpediaV2/n_val_sampled.tsv','/data/DBpediaV2/n_test_sampled.tsv', '/data/DBpediaV2/selected_train_val_ids.txt')

#df = pd.read_csv('/data/wikidata_3_class_full/train_sampled.tsv', sep='\t')
#df['PP'] = df['queryString'].apply(lambda x: check_PP(x))
#df['filter'] = df['queryString'].apply(lambda x: check_filter(x))


import os
os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
from graph_construction.jar_utils import get_query_graph,get_query_graph_nx
"""
query = """SELECT ?var1 ?var1Label WHERE { ?var1 (<http://www.wikidata.org/prop/direct/P279>)+/<http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q82955> . ?var1 <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> ; <http://www.w3.org/2000/01/rdf-schema#label> ?var1Label FILTER ( lang(?var1Label) = "en" ) }"""
"""get_query_graph_nx(query, to_dot='/PlanRGCN/qg')





base = '/data/wikidata_3_class_full/plan01_n_gq'
base = '/data/wikidata_3_class_full/'
#QueryClassAnalyzer(f"{base}/train_sampled.tsv", f"{base}/val_sampled.tsv", f"{base}/test_sampled.tsv", objective_file='/data/wikidata_3_class_full/plan01_n_gq/objective.py')


def process_tsv(path):
    import pandas as pd
    df = pd.read_csv(path, sep='\t')
    cols = [x for x in df.columns if x not in ['level_0', 'index']]
    df = df[cols]
    df.to_csv(path, sep='\t', index=False)


## Testing ged distance calculation
# create_different_k_ged_dist_matrix(basedir="/data/DBpediaV2", database_path='/data/DBpediaV2/baseline/ged.db' )



"""
exit()
"""
spl =DataSplitter(tsv_file='/data/wikidataV2/all.tsv',interval_type='percentile', percentiles=[0,50,92,100])

print('starting stratification process')
spl.make_splits_files(train_file='/data/wikidataV2/train_sampled.tsv', val_file = '/data/wikidataV2/val_sampled.tsv',
                      test_file='/data/wikidataV2/test_sampled.tsv', splt_info_file='/data/wikidataV2/queryStat.json',
                      sep='\t',
                      save_obj_file='/data/wikidataV2/data_splitter.pickle',
                      intervals_file='/data/wikidataV2/intervals.txt')
"""

# u_splt = UnseenSplitter('/data/wikidataV2/train_sampled.tsv', '/data/wikidataV2/val_sampled.tsv', '/data/wikidataV2/test_sampled.tsv','/data/wikidataV2/data_splitter.pickle','/data/wikidataV2/n_train_sampled.tsv', '/data/wikidataV2/n_val_sampled.tsv','/data/wikidataV2/n_test_sampled.tsv', '/data/wikidataV2/selected_train_val_ids.txt')

""" Need to be run
print('starting stratification process')
spl = DataSplitter(tsv_file='/data/DBpediaV2/all.tsv',interval_type='percentile', percentiles=[0,50,90,100])
spl.make_splits_files(train_file='/data/DBpediaV2/train_sampled.tsv', val_file = '/data/DBpediaV2/val_sampled.tsv',
                      test_file='/data/DBpediaV2/test_sampled.tsv', splt_info_file='/data/DBpediaV2/queryStat.json',
                      sep='\t',
                      save_obj_file='/data/DBpediaV2/data_splitter.pickle',
                      intervals_file='/data/DBpediaV2/intervals.txt')
"""
# u_splt = UnseenSplitter('/data/DBpediaV2/train_sampled.tsv', '/data/DBpediaV2/val_sampled.tsv', '/data/DBpediaV2/test_sampled.tsv','/data/DBpediaV2/data_splitter.pickle','/data/DBpediaV2/n_train_sampled.tsv', '/data/DBpediaV2/n_val_sampled.tsv','/data/DBpediaV2/n_test_sampled.tsv', '/data/DBpediaV2/selected_train_val_ids.txt')
"""
analyze = QueryClassAnalyzer('/data/wikidataV2/n_train_sampled.tsv', '/data/wikidataV2/n_val_sampled.tsv',
                             '/data/wikidataV2/n_test_sampled.tsv', '/data/wikidataV2/data_splitter.pickle')
"""
"""from graph_construction.feats.featurizer_path import FeaturizerPath

import pandas as pd
import numpy as np
from graph_construction.qp.query_plan_path import QueryPlanPath, QueryGraph
from graph_construction.query_graph import create_query_graphs_data_split
pred_stat_path='/data/planrgcn_features/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats'
ent_path='/data/planrgcn_features/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats'
lit_path='/data/planrgcn_features/extracted_features_dbpedia2016/literals/literals_stat/batches_response_stats'
featurizer = FeaturizerPath(pred_stat_path=pred_stat_path,pred_com_path=None,ent_path =ent_path, lit_path = lit_path, scaling='binner')
dgl_graphs = create_query_graphs_data_split(query_path='/data/DBpedia_3_class_full/test_sampled.tsv', is_lsq=False, feat=featurizer, query_plan=QueryGraph)

print(dgl_graphs)

exit()

from graph_construction.jar_utils import get_query_graph
query = '''PREFIX  dc: <http://purl.org/dc/elements/1.1/>
SELECT  ?title
WHERE   { <http://example.org/book/book1> dc:title* ?title }'''




df = pd.read_csv('/PlanRGCN/data/DBpedia_3_class_full/all.tsv', sep='\t')
import time

times = []
not_parse_qs = []
for i, query in enumerate(list(df['queryString'])):
    start = time.time()
    try:
        get_query_graph(query)
    except Exception:
        not_parse_qs.append(query)
    times.append(time.time()-start)
    if (i % 100) == 0:
        print(times[-100:])
        print(i, np.mean(times))

print(len(not_parse_qs), "  queries not parsed")
print(not_parse_qs)
"""
