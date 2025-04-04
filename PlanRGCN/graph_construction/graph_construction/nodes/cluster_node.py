

from graph_construction.nodes.node import Node
import numpy as np

class Cluster_node(Node):
    def __init__(self, node_label: str) -> None:
        super().__init__(node_label)
    
    def get_features(self):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features = self.get_pred_features()
        if Node.use_join_features:
            join_feat = self.get_join_features()
        else:
            join_feat = np.array([])
        return np.concatenate((nodetype ,join_feat, predicate_features))
    
    def get_pred_features(self):
        raw_predicate_freq_feats = np.zeros(1)
        cluster_bucket_feat = np.zeros(Cluster_node.max_pred_buckets)
        if self.nodetype == 1:
            raw_predicate_freq_feats[0] = self.pred_freq
            #predicate_features[1] = self.pred_literals
            #predicate_features[2] = self.pred_subject_count
            #predicate_features[3] = self.pred_object_count
            for idx in self.indices:
                cluster_bucket_feat[idx] = 1
            
        return np.concatenate( (raw_predicate_freq_feats, cluster_bucket_feat))
    
    def set_predicate_features(self):
        predicate_stat = Cluster_node.pred_feaurizer
        if predicate_stat != None:
            #self.predicate_stat = predicate_stat
            if self.type == 'URI':
                self.pred_subject_count,self.pred_object_count =-1,-1
                
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
            self.indices = predicate_stat.get_pred_feat(self.node_label)