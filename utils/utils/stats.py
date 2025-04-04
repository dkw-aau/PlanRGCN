import json
import os
import numpy as np

def load_pred_freq(file, dct, query_var="p1"):
    data = json.load(open(file, "r"))
    data = data["results"]["bindings"]
    if len(data) <= 0:
        return dct
    if not query_var in data[0].keys():
        return dct

    for x in data:
        if x[query_var]["value"] in dct.keys():
            assert x["triples"]["value"] == dct[x[query_var]["value"]]
        dct[x[query_var]["value"]] = x["triples"]["value"]
    return dct


def get_rel_dict(path):
    if not os.path.exists(path):
        raise Exception("Predicate feature not existing, "+path)
    files = sorted([f"{path}{x}" for x in os.listdir(path) if x.endswith(".json")])
    freq_dct = dict()
    for f in files:
        freq_dct = load_pred_freq(f, freq_dct)
    return freq_dct

def analyse_freq(path):
    data = get_rel_dict(path)
    vals = list(data.values())
    vals = np.array([int(x) for x in vals])
class PredStats:
    def __init__(
        self,
        path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    ) -> None:
        self.path = path
        self.triple_freq = {}
        self.pred_ents = {}
        self.pred_lits = {}
        self.pred_subj = {}
        self.pred_obj = {}
        self.load_preds_stats()
        # print(len(list(self.triple_freq.keys())))

    def load_preds_freq(self):
        freq_path = self.path + "/freq/"
        if not os.path.exists(freq_path):
            raise Exception("Predicate feature not existing")
        files = sorted(
            [f"{freq_path}{x}" for x in os.listdir(freq_path) if x.endswith(".json")]
        )
        for f in files:
            self.load_pred_freq(f)

    def load_preds_stats(self):
        freq_path = self.path + "/freq/"
        ent_path = self.path + "/ents/"
        lits_path = self.path + "/lits/"
        obj_path = self.path + "/obj/"
        subj_path = self.path + "/subj/"
        if not (
            os.path.exists(freq_path)
            and os.path.exists(ent_path)
            and os.path.exists(lits_path)
        ):
            raise Exception("Predicate feature not existing ,"+ freq_path)
        for p, f in zip(
            [freq_path, ent_path, lits_path, subj_path, obj_path],
            [
                self.load_pred_freq,
                self.load_pred_ents,
                self.load_pred_lits,
                self.load_subj,
                self.load_obj,
            ],
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
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.triple_freq.keys():
                assert x["triples"]["value"] == self.triple_freq[x["p1"]["value"]]
            self.triple_freq[x["p1"]["value"]] = x["triples"]["value"]

    def load_pred_ents(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.pred_ents.keys():
                assert x["entities"]["value"] == self.pred_ents[x["p1"]["value"]]
            self.pred_ents[x["p1"]["value"]] = x["entities"]["value"]

    def load_subj(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.pred_subj.keys():
                assert x["entities"]["value"] == self.pred_subj[x["p1"]["value"]]
            self.pred_subj[x["p1"]["value"]] = x["entities"]["value"]

    def load_obj(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if len(data) <= 0:
            return None
        if len(data) == 0:
            raise Exception(f"Reading of file contains not results: {file}")
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.pred_obj.keys():
                assert x["entities"]["value"] == self.pred_obj[x["p1"]["value"]]
            self.pred_obj[x["p1"]["value"]] = x["entities"]["value"]

    def load_pred_lits(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if data is None or len(data) == 0:
            return None
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.pred_lits.keys():
                assert x["literals"]["value"] == self.pred_lits[x["p1"]["value"]]
            self.pred_lits[x["p1"]["value"]] = x["literals"]["value"]
