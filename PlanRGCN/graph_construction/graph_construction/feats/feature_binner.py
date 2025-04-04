from graph_construction.feats.featurizer import (
    EntStats,
    LitStats,
    FeaturizerPredCoEnt,
    FeaturizerPredStats,
)
import pickle
from graph_construction.node import FilterNode, TriplePattern, TriplePattern3, FilterNode02
from graph_construction.nodes.path_node import PathNode2
from graph_construction.qp.visitor.UtilVisitor import LiteralsFeaturizer
from graph_construction.feats.feat_scale_util import BinnerEntPred
import numpy as np
from utils.stats import PredStats


class FeaturizerBinning(FeaturizerPredStats):
    def __init__(
        self,
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
        ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
        lit_path=None,
        bins=50,
        pred_end_path=None,
        scaling="None",
    ) -> None:
        self.pred_stat_path = pred_stat_path

        p = PredStats(path=pred_stat_path)
        self.pred_freq = p.triple_freq
        self.pred_ents = p.pred_ents
        self.pred_lits = p.pred_lits

        self.filter_size = 6
        #self.tp_size = 6
        # super(FeaturizerPredStats,self).__init__(pred_stat_path)

        self.pred2index, self.max_pred = pickle.load(open(pred_com_path, "rb"))
        estat = EntStats(path=ent_path)
        self.ent_freq = estat.ent_freq
        self.ent_subj = estat.subj_ents
        self.ent_obj = estat.obj_ents
        
        self.lit_path = lit_path
        if lit_path is not None:
            lstat = LitStats(path=lit_path)
            self.lit_freq = lstat.lits
        else:
            self.lit_freq = None
        

        self.scaling = "binner"
        self.scaler = BinnerEntPred(
            self.ent_freq,
            self.ent_subj,
            self.ent_obj,
            self.pred_freq,
            self.pred_ents,
            self.pred_lits,
            self.lit_freq,
            bins=bins,
        )
        if lit_path is not None:
            self.tp_size = (
                self.max_pred
                + 3
                + self.scaler.ent_scale_len() * 2
                + self.scaler.pred_scale_len()
                + self.scaler.lit_scale_len()
                + LiteralsFeaturizer.feat_size()
            )
        else:
            self.tp_size = (
                self.max_pred
                + 3
                + self.scaler.ent_scale_len() * 2
                + self.scaler.pred_scale_len()
            )

    def featurize(self, node):
        if isinstance(node, FilterNode02):
            return self.filter_features(node).astype("float32")
        elif isinstance(node, TriplePattern3) or isinstance(node, PathNode2):
            return np.concatenate(
                (self.tp_features(node), self.pred_clust_features(node)), axis=0
            ).astype("float32")
        else:
            raise Exception("unknown node type"+ str(type(node)))

    def pred_clust_features(self, node: TriplePattern):
        vec = np.zeros(self.max_pred)
        try:
            idx = self.pred2index[node.predicate.node_label]
        except KeyError:
            idx = self.max_pred - 1
        if isinstance(idx, list):
            for i in idx:
                vec[i] = 1
        else:
            vec[idx] = 1
        return vec

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )
        freq = self.get_value_dict(self.pred_freq, node.predicate.node_label)
        lits = self.get_value_dict(self.pred_lits, node.predicate.node_label)
        ents = self.get_value_dict(self.pred_ents, node.predicate.node_label)

        freq, lits, ents = self.scaler.pred_scale(freq, lits, ents)

        (
            subj_freq,
            subj_subj_freq,
            sub_obj_freq,
            obj_freq,
            obj_subj_freq,
            obj_obj_freq,
        ) = (0, 0, 0, 0, 0, 0)

        if node.subject.type == "URI":
            subj_freq = self.get_value_dict(self.ent_freq, node.subject.node_label)
            subj_subj_freq = self.get_value_dict(self.ent_subj, node.subject.node_label)
            sub_obj_freq = self.get_value_dict(self.ent_obj, node.subject.node_label)
            subj_freq, subj_subj_freq, sub_obj_freq = self.scaler.ent_scale(
                subj_freq, subj_subj_freq, sub_obj_freq
            )
        else:
            subj_freq, subj_subj_freq, sub_obj_freq = self.scaler.ent_scale_no_values()

        if node.object.type == "URI":
            obj_freq = self.get_value_dict(self.ent_freq, node.subject.node_label)
            obj_subj_freq = self.get_value_dict(self.ent_subj, node.subject.node_label)
            obj_obj_freq = self.get_value_dict(self.ent_obj, node.subject.node_label)
            obj_freq, obj_subj_freq, obj_obj_freq = self.scaler.ent_scale(
                obj_freq, obj_subj_freq, obj_obj_freq
            )
            if self.lit_path is not None:
                lit_freq_o = self.scaler.lit_scale_no_values()
                lit_type_la = LiteralsFeaturizer.feat(node.object)
        elif node.object.type == "LIT":
            obj_freq, obj_subj_freq, obj_obj_freq = self.scaler.ent_scale_no_values()
            if self.lit_path is not None:
                lit_freq_o = self.get_value_dict(self.lit_freq, node.object.node_label)
                lit_freq_o = self.scaler.lit_scale(lit_freq_o)
                lit_type_la = LiteralsFeaturizer.feat(node.object)
        else:
            obj_freq, obj_subj_freq, obj_obj_freq = self.scaler.ent_scale_no_values()
            if self.lit_path is not None:
                lit_type_la = LiteralsFeaturizer.feat(node.object)
                lit_freq_o = self.scaler.lit_scale_no_values()

        if self.lit_path is None:
            stat_vec = np.concatenate(
                [
                    freq,
                    lits,
                    ents,
                    subj_freq,
                    subj_subj_freq,
                    sub_obj_freq,
                    obj_freq,
                    obj_subj_freq,
                    obj_obj_freq,
                ])
        else:
            stat_vec = np.concatenate(
                [
                    freq,
                    lits,
                    ents,
                    subj_freq,
                    subj_subj_freq,
                    sub_obj_freq,
                    obj_freq,
                    obj_subj_freq,
                    obj_obj_freq,
                    lit_freq_o,
                    lit_type_la,
                ])
        return np.concatenate((var_vec, stat_vec, np.zeros(self.filter_size)), axis=0)
