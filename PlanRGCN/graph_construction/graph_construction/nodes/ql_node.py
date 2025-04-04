

#Training query log features for this node class 
# use the feature_extraction.predicates.ql_pred_featurizer for this node.

from graph_construction.nodes.node import Node
from feature_extraction.predicates.ql_pred_featurizer import ql_pred_featurizer
import numpy as np

class ql_node(Node):
    def __init__(self, node_label: str) -> None:
        super().__init__(node_label)
        
    def get_pred_features(self):
        predicate_stat = ql_node.pred_feaurizer
        raw_pred_features = np.zeros(4)
        topk_features = np.zeros(predicate_stat.freq_k)
        bin_features = np.zeros(predicate_stat.total_bin+1)
        
        if self.nodetype == 1:
            
            raw_pred_features[0] = self.pred_freq
            raw_pred_features[1] = self.pred_literals
            raw_pred_features[2] = self.pred_subject_count
            raw_pred_features[3] = self.pred_object_count
            
            topk_features[self.topk] = 1
            bin_features[self.bin] = 1
        
        return np.concatenate((raw_pred_features, topk_features, bin_features))
    
    def set_predicate_features(self):
        predicate_stat = ql_node.pred_feaurizer
        if predicate_stat == None or (not isinstance(predicate_stat,ql_pred_featurizer)):
            raise RuntimeError("Predicate Featurizer has not been setup properly!")
            #self.predicate_stat = predicate_stat
        if self.type == 'URI':
            self.bin = predicate_stat.get_bin(self.node_label)
            self.topk = predicate_stat.top_k_predicate(self.node_label)
            
            if self.node_label in predicate_stat.predicate_freq.keys():
                    self.pred_freq = predicate_stat.predicate_freq[self.node_label]
            else:
                self.pred_freq = -1        
                
            if self.node_label in predicate_stat.uniqueLiteralCounter.keys():
                self.pred_literals = predicate_stat.uniqueLiteralCounter[self.node_label]
            else:
                self.pred_literals = -1
                
            
            if self.node_label in predicate_stat.unique_entities_counter.keys():
                self.pred_subject_count,self.pred_object_count = predicate_stat.unique_entities_counter[self.node_label]
            else:
                self.pred_subject_count = -1
                self.pred_object_count = -1