import pandas as pd
from q_gen.util import Utility
pred_stat_path = '/data/metaKGStat/dbpedia/predicate/pred_stat/batches_response_stats/freq'
pred_freq = Utility.get_pred_freq(pred_stat_path)
train_log = '/data/DBpedia_3_class_full/train_sampled.tsv'
val_log = '/data/DBpedia_3_class_full/val_sampled.tsv'
vql = pd.read_csv(val_log, sep='\t')
tql = pd.read_csv(train_log, sep='\t')
train_val_rels, train_val_ents = Utility.get_ent_rels_from_train_val(tql, vql)


def r_c(rel):
    global train_val_rels
    if rel in train_val_rels:
        return True
    else :
        return False