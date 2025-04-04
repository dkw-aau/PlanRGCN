import json
import os
import pickle
import numpy as np
from graph_construction.node import Node, FilterNode, TriplePattern
from scalers import (
    EntDefaultScaler,
    EntMinMaxScaler,
    EntStandardScaler,
    EntStandardScalerPredSubjObj,
    LogScaler,
)
from sklearn.preprocessing import RobustScaler
from utils.stats import PredStats


class FeaturizerBase:
    def __init__(self, vec_size) -> None:
        self.vec_size = vec_size

    def featurize(self, node):
        return np.array([1, 0, 0, 0, 0])


class FeaturizerPredStats(FeaturizerBase):
    def __init__(
            self,
            pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    ) -> None:
        self.pred_stat_path = pred_stat_path

        p = PredStats(path=pred_stat_path)
        self.pred_freq = p.triple_freq
        self.pred_ents = p.pred_ents
        self.pred_lits = p.pred_lits

        self.filter_size = 6
        self.tp_size = 6

    def featurize(self, node):
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float32")
        elif isinstance(node, TriplePattern):
            return self.tp_features(node).astype("float32")
        else:
            raise Exception("unknown node type")

    def filter_features(self, node):
        # vec = np.zeros(6)
        vec = list()
        for f in [
            FilterFeatureUtils.isLogical,
            FilterFeatureUtils.isArithmetic,
            FilterFeatureUtils.isComparison,
            FilterFeatureUtils.isGeneralFunction,
            FilterFeatureUtils.isStringManipulation,
            FilterFeatureUtils.isTime,
        ]:
            if f(node.expr_string):
                vec.append(1)
            else:
                vec.append(0)
        return np.concatenate((np.zeros(self.tp_size), np.array(vec)), axis=0)

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )
        freq = self.get_value_dict(self.pred_freq, node.predicate.node_label)
        lits = self.get_value_dict(self.pred_lits, node.predicate.node_label)
        ents = self.get_value_dict(self.pred_ents, node.predicate.node_label)

        stat_vec = np.array([freq, lits, ents])
        return np.concatenate((var_vec, stat_vec, np.zeros(self.filter_size)), axis=0)

    def get_value_dict(self, dct: dict, predicate: str):
        try:
            if predicate.startswith('<'):
                predicate = predicate[1:]
            if predicate.endswith('>'):
                predicate = predicate[:-1]
            return dct[predicate]
        except KeyError:
            return 0


class FeaturizerPredCo(FeaturizerPredStats):
    def __init__(
            self,
            pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
            pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    ) -> None:
        super().__init__(pred_stat_path)

        self.pred2index, self.max_pred = pickle.load(open(pred_com_path, "rb"))
        self.tp_size = self.tp_size + self.max_pred

    def featurize(self, node):
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float64")
        elif isinstance(node, TriplePattern):
            return np.concatenate(
                (self.tp_features(node), self.pred_clust_features(node)), axis=0
            ).astype("float64")
        else:
            raise Exception("unknown node type")

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


class FeaturizerPredCoEnt(FeaturizerPredStats):
    def __init__(
            self,
            pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
            pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
            ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
            pred_end_path=None,
            scaling="None",
    ) -> None:
        super().__init__(pred_stat_path)

        self.pred2index, self.max_pred = pickle.load(open(pred_com_path, "rb"))
        self.tp_size = self.tp_size + self.max_pred + 6
        estat = EntStats(path=ent_path)
        self.ent_freq = estat.ent_freq
        self.ent_subj = estat.subj_ents
        self.ent_obj = estat.obj_ents

        self.scaling = scaling
        match self.scaling:
            case "min-max":
                self.scaler = EntMinMaxScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                )
            case "std":
                self.scaler = EntStandardScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                )
            case "robust":
                self.scaler = EntDefaultScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                    scale_class=RobustScaler,
                )
            case "log":
                self.scaler = EntDefaultScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                    scale_class=RobustScaler,
                )
                """self.scaler = LogScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                )"""
            case _:
                raise Exception(
                    f"Scaling option {scaling} is undefined! Either implement is or use a predefined one."
                )

    def featurize(self, node):
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float32")
        elif isinstance(node, TriplePattern):
            return np.concatenate(
                (self.tp_features(node), self.pred_clust_features(node)), axis=0
            ).astype("float32")
        else:
            raise Exception("unknown node type")

    def pred_clust_features(self, node: TriplePattern):
        vec = np.zeros(self.max_pred)
        try:
            idx = self.pred2index[node.predicate.node_label]
        except KeyError:
            idx = self.max_pred - 1
        vec[idx] = 1
        return vec

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )
        freq = self.get_value_dict(self.pred_freq, node.predicate.node_label)
        lits = self.get_value_dict(self.pred_lits, node.predicate.node_label)
        ents = self.get_value_dict(self.pred_ents, node.predicate.node_label)
        if not self.scaling == "None":
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
            if not self.scaling == "None":
                subj_freq, subj_subj_freq, sub_obj_freq = self.scaler.ent_scale(
                    subj_freq, subj_subj_freq, sub_obj_freq
                )

        if node.object.type == "URI":
            obj_freq = self.get_value_dict(self.ent_freq, node.subject.node_label)
            obj_subj_freq = self.get_value_dict(self.ent_subj, node.subject.node_label)
            obj_obj_freq = self.get_value_dict(self.ent_obj, node.subject.node_label)
            if not self.scaling == "None":
                obj_freq, obj_subj_freq, obj_obj_freq = self.scaler.ent_scale(
                    obj_freq, obj_subj_freq, obj_obj_freq
                )

        stat_vec = np.array(
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
            ]
        )
        return np.concatenate((var_vec, stat_vec, np.zeros(self.filter_size)), axis=0)


class FeaturizerSubjPred(FeaturizerPredCoEnt):
    """Featurizes nodes based on triple pattern node or filter node.
    The triple pattern features are augmented with no of literal and object and subject.
    """

    def __init__(
            self,
            pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
            pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
            ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
            pred_end_path="/PlanRGCN/extracted_features_dbpedia2016/pred_ent/batch_response",
            scaling="None",
    ) -> None:
        super().__init__(pred_stat_path, pred_com_path, ent_path, scaling)
        if self.scaling == "min-max":
            self.scaler = EntMinMaxScaler(
                self.ent_freq,
                self.ent_subj,
                self.ent_obj,
                self.pred_freq,
                self.pred_ents,
                self.pred_lits,
            )
        if self.scaling == "std":
            self.pred_extra_scaler = EntStandardScalerPredSubjObj()
            self.scaler = EntStandardScaler(
                self.ent_freq,
                self.ent_subj,
                self.ent_obj,
                self.pred_freq,
                self.pred_ents,
                self.pred_lits,
            )


class FilterFeatureUtils:
    def isLogical(filter_expr):
        lst = ["||", "&&"]  # '!' can be confused with '!=' comparison operator
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isArithmetic(filter_expr):
        lst = ["*", "+", "/", "-"]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isComparison(filter_expr):
        lst = ["=", "!=", "<", ">", "<=", ">="]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isGeneralFunction(filter_expr):
        lst = [
            "DATATYPE",
            "STR",
            "IRI",
            "LANG",
            "BOUND",
            "IN",
            "NOT IN",
            "isBlank",
            "isIRI",
            "isLiteral",
        ]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isStringManipulation(filter_expr):
        lst = [
            "STRLEN",
            "SUBSTR",
            "UCASE",
            "LCASE",
            "STRSTARTS",
            "STRENDS",
            "CONTAINS",
            "STRBEFORE",
            "STRAFTER",
            "ENCODE_FOR_URI",
            "CONCAT",
            "LANGMATCHES",
            "REGEX",
            "REPLACE",
        ]

        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isTime(filter_expr):
        lst = [
            "NOW",
            "YEAR",
            "MONTH",
            "DAY",
            "HOURS",
            "MINUTES",
            "SECONDS",
            "TIMEZONE",
            "TZ",
        ]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isSubstring(filter_expr, lst):
        filter_expr = filter_expr.lower()
        lst = [x.lower() for x in lst]
        for x in lst:
            if x in filter_expr:
                return True
        return False


# p = PredStats()
# print(len(list(p.pred_ents.keys())))
# print(len(list(p.triple_freq.keys())))
# print(len(list(p.pred_lits.keys())))
class EntStats:
    def __init__(
            self,
            path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
    ) -> None:
        self.path = path
        self.ent_freq = {}
        self.obj_ents = {}
        self.subj_ents = {}
        self.load_ent_stats()
        # print(len(list(self.triple_freq.keys())))

    def load_preds_freq(self):
        freq_path = self.path + "/freq/"
        if not os.path.exists(freq_path):
            raise Exception("Entity feature not existing")
        files = sorted(
            [f"{freq_path}{x}" for x in os.listdir(freq_path) if x.endswith(".json")]
        )
        for f in files:
            self.load_pred_freq(f)

    def load_ent_stats(self):
        freq_path = self.path + "/freq/"
        subj_path = self.path + "/subj/"
        obj_path = self.path + "/obj/"
        if not (
                os.path.exists(freq_path)
                and os.path.exists(subj_path)
                and os.path.exists(obj_path)
        ):
            raise Exception("Predicate feature not existing " + self.path)
        for p, f in zip(
                [freq_path, subj_path, obj_path],
                [self.load_pred_freq, self.load_subj_ents, self.load_obj_ents],
        ):
            self.load_preds_stat_helper(p, f)

    def load_preds_stat_helper(self, path, loader_func):
        files = sorted([f"{path}{x}" for x in os.listdir(path) if x.endswith(".json")])
        for f in files:
            loader_func(f)

    def load_pred_freq(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None

        if not "e" in data[0].keys():
            return None

        for x in data:
            if x["e"]["value"] in self.ent_freq.keys():
                assert x["entities"]["value"] == self.ent_freq[x["e"]["value"]]
            self.ent_freq[x["e"]["value"]] = x["entities"]["value"]

    def load_subj_ents(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None

        if not "e" in data[0].keys():
            return None

        for x in data:
            if x["e"]["value"] in self.subj_ents.keys():
                assert x["entities"]["value"] == self.subj_ents[x["e"]["value"]]
            self.subj_ents[x["e"]["value"]] = x["entities"]["value"]

    def load_obj_ents(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None

        if not "e" in data[0].keys():
            return None

        for x in data:
            if x["e"]["value"] in self.obj_ents.keys():
                assert x["entities"]["value"] == self.obj_ents[x["e"]["value"]]
            self.obj_ents[x["e"]["value"]] = x["entities"]["value"]


class LitStats:
    def __init__(
            self,
            path=None,
    ) -> None:
        if path is None:
            raise Exception("path must be specified")

        self.path = path
        self.lits = {}
        self.load_lit_freqs()
        # print(len(list(self.triple_freq.keys())))

    def load_lit_freqs(self):
        freq_path = self.path + "/freq/"
        if not os.path.exists(freq_path):
            raise Exception("Literal feature not existing " + freq_path)
        files = sorted(
            [f"{freq_path}{x}" for x in os.listdir(freq_path) if x.endswith(".json")]
        )
        for f in files:
            self.load_lit_freq(f)

    def load_lit_freq(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None

        if not "e" in data[0].keys():
            return None

        for x in data:
            if x["e"]["value"] in self.lits.keys():
                if x["entities"]["value"] > self.lits[x["e"]["value"]]:
                    self.lits[x["e"]["value"]] = x["entities"]["value"]
            else:
                self.lits[x["e"]["value"]] = x["entities"]["value"]
