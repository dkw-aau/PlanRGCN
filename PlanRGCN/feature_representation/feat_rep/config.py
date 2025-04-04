config = {
        'save_dir' : '/PlanRGCN/data/dbpedia2016/predicate/pred_co',
        'pred_co_pair_dir' : '/PlanRGCN/data/dbpedia2016/predicate/batch_response/',
}
config["comm"] = f"{config['save_dir']}/communities_louvain.pickle"
config["p2i"] = f"{config['save_dir']}/pred2index_louvain.pickle"
config["time_log"] = f"{config['save_dir']}/time_pred_co_louvain.txt"
