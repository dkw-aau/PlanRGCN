"""
The goal of this script is to compute the predicate community features to be used by the different datasets.
"""
import sys

from feat_rep.pred.pred_co import (
    PredicateCommunityCreator,
    create_kernighan_lin,
    create_louvain_to_p_index,
)


# General Paths
pred_co_path = sys.argv[1]
pred_co_response = sys.argv[2]
com_path = f"{pred_co_path}/communities_louvain.pickle"
pred_graph_path = f"{pred_co_path}/pred_graph.pickle"
louvain_com_path = f"{pred_co_path}/pred2index_louvain.pickle"
louvain_com_path = f"{pred_co_path}/pred2index_louvain2.pickle"

# Wikidata Paths
#pred_co_path = "/PlanRGCN/extracted_features_wd/predicate/pred_co"
#pred_co_response = (
#    "/PlanRGCN/extracted_features_wd/predicate/predicate_cooccurence/batch_response/"
#)
#com_path = f"{pred_co_path}/communities_louvain.pickle"
#pred_graph_path = f"{pred_co_path}/pred_graph.pickle"
#louvain_com_path = f"{pred_co_path}/pred2index_louvain.pickle"
#louvain_com_path = f"{pred_co_path}/pred2index_louvain2.pickle"

# DBpedia 2016 Paths
#pred_co_path = "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co"
#pred_co_response = "/PlanRGCN/extracted_features_dbpedia2016/predicate/predicate_cooccurence/batch_response/"
#pred_graph_path = f"{pred_co_path}/pred_graph.pickle"
# com_path = f"{pred_co_path}/communities_louvain2.pickle"
# louvain_com_path = f"{pred_co_path}/pred2index_louvain2.pickle"

d = PredicateCommunityCreator(save_dir=pred_co_path)
d.get_louvain_communities(dir=pred_co_response, save_pred_graph=pred_graph_path)
create_louvain_to_p_index(
    path=com_path,
    output_path=louvain_com_path,
)

## this one took too long for DBpedia KG.
#create_kernighan_lin(
#    pred_graph_path=pred_graph_path,
#    iterations=5,
#    save_dict_path=f"{pred_co_path}/pred2index_pred2_kernighan.pickle",
#)
