import numpy as np


class FilterNode:
    def __init__(self, data) -> None:
        self.data = data
        self.expr_string = data["expr"]
        splts = data["expr"].split(" ")
        self.vars = [x for x in splts if x.startswith("?")]
        for i in range(len(self.vars)):
            if ")" in self.vars[i]:
                self.vars[i] = self.vars[i].split(")")[0]

    def __str__(self) -> str:
        return f"Filter node : {self.expr_string}"

    def __hash__(self) -> int:
        return self.data["expr"].__hash__()

class FilterNode02(FilterNode):
    def __init__(self, data) -> None:
        self.data = data
        self.expr_string = data["expression"]
        self.vars = data['vars']
        """for i in range(len(self.vars)):
            if ")" in self.vars[i]:
                self.vars[i] = self.vars[i].split(")")[0]"""


class Node:
    pred_bins = 30
    pred_topk = 15
    pred_feat_sub_obj_no: bool = None
    use_ent_feat: bool = False
    ent_bins: int = None
    use_join_features: bool = True

    def __init__(self, node_label: str) -> None:
        self.node_label = node_label
        if node_label.startswith("?") or node_label.startswith("$"):
            self.type = "VAR"
        elif node_label.startswith("<http") or node_label.startswith("http"):
            self.type = "URI"
        elif node_label.startswith("join"):
            self.type = "JOIN"
        elif node_label.startswith("_:") or node_label.startswith(":"):
            # self.type = "BLANK"
            self.type = "VAR"  # in graph pattern acts as variables
        else:
            self.type = "LIT"

        self.pred_freq = None
        self.pred_literals = None
        self.pred_entities = None
        # self.topK = None

        # for join node
        self.is_subject_var = None
        self.is_pred_var = None
        self.is_object_var = None

    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f"{self.type} {self.node_label}"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.node_label == other
        return self.node_label == other.node_label

    def __hash__(self) -> int:
        return hash(self.node_label)

    def get_pred_features(self):
        pred_bins, pred_topk, pred_feat_sub_obj_no = (
            Node.pred_bins,
            Node.pred_topk,
            Node.pred_feaurizer,
        )

        predicate_bins = np.zeros(pred_bins)
        topk_vec = np.zeros(pred_topk)

        if pred_feat_sub_obj_no:
            predicate_features = np.zeros(4)
        else:
            predicate_features = np.zeros(3)
        if self.nodetype == 1:
            predicate_features[0] = self.pred_freq
            predicate_features[1] = self.pred_literals
            if not pred_feat_sub_obj_no:
                predicate_features[2] = self.pred_entities
            else:
                predicate_features[2] = self.pred_subject_count
                predicate_features[3] = self.pred_object_count
            try:
                predicate_bins[self.bucket] = 1
            except AttributeError:
                predicate_bins[pred_bins - 1] = 1
            if self.topK != None and self.topK < pred_topk:
                topk_vec[self.topK] = 1
        if np.sum(np.isnan(predicate_features)) > 0:
            predicate_features[np.isnan(predicate_features)] = 0
            # raise Exception
        if np.sum(np.isnan(predicate_bins)) > 0:
            raise Exception
        if np.sum(np.isnan(topk_vec)) > 0:
            raise Exception
        return np.concatenate((predicate_features, predicate_bins, topk_vec))
        # return predicate_features,predicate_bins, topk_vec

    def get_ent_features(self, ent_bins):
        freq_vec_ent = np.zeros(1)
        ent_bins_vec = np.zeros(ent_bins + 1)
        if self.nodetype in [0, 2] and self.type == "URI":
            freq_vec_ent[0] = self.ent_freq
            ent_bins_vec[self.ent_bin] = 1
        if np.sum(np.isnan(freq_vec_ent)) > 0:
            raise Exception
        if np.sum(np.isnan(ent_bins_vec)) > 0:
            raise Exception
        return np.concatenate((freq_vec_ent, ent_bins_vec))

    def get_join_features(self):
        join_feat = np.zeros(3)
        if self.nodetype == 3:
            return join_feat
        if self.is_subject_var:
            join_feat[0] = 1
        if self.is_pred_var:
            join_feat[1] = 1
        if self.is_object_var:
            join_feat[2] = 1
        return join_feat

    def get_features(self):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features = self.get_pred_features()
        if Node.use_ent_feat:
            ent_features = self.get_ent_features(Node.ent_bins)
        else:
            ent_features = np.array([])
        if Node.use_join_features:
            join_feat = self.get_join_features()
        else:
            join_feat = np.array([])
        return np.concatenate((nodetype, join_feat, predicate_features, ent_features))

    def set_predicate_features(self):
        self.pred_freq = -1
        self.pred_literals = -1
        self.pred_subject_count, self.pred_object_count = -1, -1


def is_variable_check(label: str):
    label = label.strip()
    if label.startswith("?") or label.startswith("$"):
        return True
    return False


class TriplePattern:
    """Class representing a triple pattern. Joins on constants are not considered separately"""

    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class

        self.subject = node_class(data["Subject"])
        self.predicate = node_class(data["Predicate"])
        self.object = node_class(data["Object"])
        # with good results of 80% f1 score - old encoding
        self.subject.nodetype = 0
        self.predicate.nodetype = 1
        self.object.nodetype = 2

        # New variable encoding
        self.subject.nodetype = 0 if is_variable_check(self.subject.node_label) else 1
        self.predicate.nodetype = (
            0 if is_variable_check(self.predicate.node_label) else 1
        )
        self.object.nodetype = 0 if is_variable_check(self.object.node_label) else 1

        if "level" in data.keys():
            self.level = data["level"]

    def __hash__(self):
        return hash(self.subject) + hash(self.predicate) + hash(self.object)

    def __str__(self):
        return f"Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"

    def __repr__(self):
        return f"Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"

    def __eq__(self, other):
        if isinstance(other, dict):
            return (
                self.subject.node_label == other["Subject"]
                and self.predicate.node_label == other["Predicate"]
                and self.object.node_label == other["Object"]
            )
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
        )

    def get_variables(self):
        v = []
        v.append(self.subject)
        v.append(self.predicate)
        v.append(self.object)
        """if self.subject.type == "VAR":
            v.append(self.subject)
        if self.predicate.type == "VAR":
            v.append(self.predicate)
        if self.object.type == "VAR":
            v.append(self.object)"""
        return v

    def get_joins(self):
        return list(set(self.get_variables()))
class TriplePattern2(TriplePattern):
    """Class representing a triple pattern. Joins on constants are not considered separately"""

    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class

        self.subject = node_class(data["Subject"])
        self.predicate = node_class(data["Predicate"])
        self.object = node_class(data["Object"]["value"])
        try:
            self.object.datatype = data["Object"]["datatype"]
        except Exception:
            pass
        try:
            self.object.langtag = data["Object"]["langTag"]
        except Exception:
            pass

        # with good results of 80% f1 score - old encoding
        self.subject.nodetype = 0
        self.predicate.nodetype = 1
        self.object.nodetype = 2

        # New variable encoding
        self.subject.nodetype = 0 if is_variable_check(self.subject.node_label) else 1
        self.predicate.nodetype = (
            0 if is_variable_check(self.predicate.node_label) else 1
        )
        self.object.nodetype = 0 if is_variable_check(self.object.node_label) else 1

        if "level" in data.keys():
            self.level = data["level"]

class TriplePattern3(TriplePattern2):
    """Class representing a triple pattern. Joins on constants are not considered separately"""

    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class

        self.subject = node_class(data["subject"])
        self.predicate = node_class(data["predicate"])
        self.object = node_class(data["object"])
        if data['isLiteral']:
            self.object.datatype = data['objectDatatype']
            try:
                if data['objectLang'] != "":
                    self.object.langtag = data["objectLang"]
            except Exception:
                pass

        self.subject.nodetype = 0
        self.predicate.nodetype = 1
        self.object.nodetype = 2

        # New variable encoding
        self.subject.nodetype = 0 if is_variable_check(self.subject.node_label) else 1
        self.predicate.nodetype = (
            0 if is_variable_check(self.predicate.node_label) else 1
        )
        self.object.nodetype = 0 if is_variable_check(self.object.node_label) else 1