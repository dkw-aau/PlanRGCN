

from graph_construction.nodes.node import Node
import numpy as np

class Node_num_pred_encoding(Node):
    def __init__(self, node_label: str) -> None:
        super().__init__(node_label)
        
    
    def get_pred_features(self):
        predicate_features = np.zeros(2)
        if self.nodetype == 1:
            predicate_features[0] = self.pred_freq
            #predicate_features[1] = self.pred_literals
            #predicate_features[2] = self.pred_subject_count
            #predicate_features[3] = self.pred_object_count
            predicate_features[1] = self.predicate_id
        
        return predicate_features
    
    def set_predicate_features(self):
        predicate_stat = Node_num_pred_encoding.pred_feaurizer
        if predicate_stat != None:
            #self.predicate_stat = predicate_stat
            if self.type == 'URI':
                self.pred_subject_count,self.pred_object_count =-1,-1
                self.predicate_id = int(predicate_stat.get_pred_feat(self.node_label))
                
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
                
                #self.pred_freq = predicate_stat.get_pred_feat2(self.node_label)