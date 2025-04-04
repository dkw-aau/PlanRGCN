from feat_con_time.time_loader import TimeLoader
import numpy as np

def extract_processing_times(config):
    tm_data = {
                'pred_stat': {
                    'ents':[],
                    'lits':[],
                    'freq':[],
                    'subj':[],
                    'obj':[],
                    'pred_co_extract': [],
                },
                'ent_stat': {
                    'freq':[],
                    'subj':[],
                    'obj':[],
                },
                'lit_stat': {
                    'freq':[],
                },
    }
    for i in config['pred_stat']:
        tm_data = TimeLoader.load_pred_stat(i, tm_data)
    for i in config['pred_obj_subj']:
        tm_data = TimeLoader.load_pred_stat(i, tm_data)
    for i in config['pred_extract']:
        tm_data = TimeLoader.load_pred_extract(i, tm_data)
    for i in config['pred_co_extract']:
        tm_data = TimeLoader.load_pred_co_extract(i, tm_data)
    for i in config['pred_louvain']:
        tm_data = TimeLoader.load_pred_louvain(i, tm_data)
        
    for i in config['ent_stat']:
        tm_data = TimeLoader.load_ent_stat(i, tm_data)
    for i in config['ent_extract']:
        tm_data = TimeLoader.load_ent_extract(i, tm_data)
        
    for i in config['lit_stat']:
        tm_data = TimeLoader.load_lit_stat(i, tm_data)
    for i in config['lit_extract']:
        tm_data = TimeLoader.load_lit_extract(i, tm_data)
    
    return tm_data

def replace_key(data, old_key, new_key):
    data[new_key] = data[old_key]
    del data[old_key]
    return data
    
def rename_keys(res):
    res = replace_key(res, 'pred_stat_ents', '# of entities for each predicates')
    res = replace_key(res, 'pred_stat_subj', '# of subjects for each predicates')
    res = replace_key(res, 'pred_stat_obj', '# of objects for each predicates')
    res = replace_key(res, 'pred_stat_lits', '# of literals for each predicates')
    res = replace_key(res, 'pred_stat_freq', '# of triples for each predicates')
    res = replace_key(res, 'pred_stat_pred_co_extract', 'Predicate Co-occurrence pairs')
    res = replace_key(res, 'ent_stat_freq', '# of triples for each entity')
    res = replace_key(res, 'ent_stat_subj', '# of subjects for each entity')
    res = replace_key(res, 'ent_stat_obj', '# of objects for each entity')
    res = replace_key(res, 'lit_stat_freq', '# of triples for each literal')
    return res
    
def summation(data):
    res = {}
    for k in data['pred_stat']:
        res[f"pred_stat_{k}"] = np.sum(data['pred_stat'][k])
    for k in data['ent_stat']:
        res[f"ent_stat_{k}"] = np.sum(data['ent_stat'][k])
    for k in data['lit_stat']:
        res[f"lit_stat_{k}"] = np.sum(data['lit_stat'][k])
    res = rename_keys(res)
    res['Predicate Extraction'] = data['pred_extract']
    res['Louvain Community Detection'] = data['Louvain Communication'] +data['pred2index']
    res['Entity Extraction'] = data['ent_extract']
    res['Literal  Extraction'] = data['lit_extract']
    return res


def get_formatted_time(seconds):
    hours = seconds // (60*60)
    seconds = seconds % (60*60)
    minutes = seconds // 60
    seconds = round(seconds % 60,2)
    return hours, minutes, seconds
    
def get_rel_time(sum_data):
    perc_data = {}
    total = 0
    for k in sum_data.values():
        total+= k
    for k in sum_data.keys():
        perc_data[k] = (sum_data[k]/total)*100
    return perc_data, total


def get_time_data(*args):
    data = {}
    for config in args:
        time_data = extract_processing_times(config)
        sum_data = summation(time_data)
        data[config['name']] = sum_data
        perc_data = get_rel_time(sum_data)
        print("Total time %s : %sh %sm %ss"%(config['name'], *get_formatted_time(perc_data[1])))
    return data