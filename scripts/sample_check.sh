queryplandir=/PlanRGCN/extracted_features/queryplans/
split_dir=/qpp/dataset/DBpedia_2016_12k_simple_opt_filt
#split_dir=/qpp/dataset/DBpedia_2016_12k_sample_simple/
pred_stat_path=/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats
pred_com_path=/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle
ent_path=/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats
scaling=\"None\"


train_path=\"$split_dir/train_sampled.tsv\"
val_path=\"$split_dir/val_sampled.tsv\"
test_path=\"$split_dir/test_sampled.tsv\"

python3 -c """
from sample_checker.chk import input_check
query_plan_dir='/PlanRGCN/extracted_features/queryplans/'
query_path='/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/train_sampled.tsv'
pred_stat_path='/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats'
pred_com_path='/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle'
ent_path='/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats'
gen = input_check(query_plan_dir,query_path,pred_stat_path, pred_com_path, ent_path, scaling, is_lsq=True)

sample = next(gen)
"""