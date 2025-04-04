from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np


class Scaler:
    def pred_scale_len(self):
        return 3
    def ent_scale_len(self):
        return 3
    def ent_scale_no_values(self):
        ###maybe should be 0
        return np.array([-1]),np.array([-1]),np.array([-1])

class LogScaler(Scaler):
    def __init__(
        self, ent_freq, ent_subj, ent_obj, pred_freq, pred_ents, pred_lits
    ) -> None:
        pass

    def pred_scale(self, freq, lits, ents):
        freq = np.log(freq) if freq != 0 else freq
        try:
            lits = np.log(lits) if lits != 0 else lits
        except  TypeError:
            raise Exception("Did not work for", freq)
        try:
            ents = np.log(ents) if ents != 0 else ents
        except  TypeError:
            raise Exception("Did not work for", freq)
        return freq, lits, ents

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = np.log(ent_freq) if ent_freq != 0 else ent_freq
        subj_freq = np.log(subj_freq) if subj_freq != 0 else subj_freq
        obj_freq = np.log(obj_freq) if obj_freq != 0 else obj_freq
        
        return ent_freq, subj_freq, obj_freq
    
class EntMinMaxScaler(Scaler):
    def __init__(
        self, ent_freq, ent_subj, ent_obj, pred_freq, pred_ents, pred_lits
    ) -> None:
        # predicate frequency
        self.pred_freq_scaler = MinMaxScaler()
        input_lst = np.array([int(x) for x in pred_freq.values()]).reshape(-1, 1)
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array([int(x) for x in pred_lits.values()]).reshape(-1, 1)
        self.pred_lits_scaler = MinMaxScaler()
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array([int(x) for x in pred_ents.values()]).reshape(-1, 1)
        self.pred_ents_scaler = MinMaxScaler()
        self.pred_ents_scaler.fit(input_lst)

        # Entity Frequency
        input_lst = np.array([int(x) for x in ent_freq.values()]).reshape(-1, 1)
        self.ent_freq_scaler = MinMaxScaler()
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array([int(x) for x in ent_subj.values()]).reshape(-1, 1)
        self.ent_sub_scaler = MinMaxScaler()
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array([int(x) for x in ent_obj.values()]).reshape(-1, 1)
        self.ent_obj_scaler = MinMaxScaler()
        self.ent_obj_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform([[freq]])[0]
        lits = self.pred_lits_scaler.transform([[lits]])[0]
        ents = self.pred_ents_scaler.transform([[ents]])[0]
        # return np.log(freq), np.log(lits), np.log(ents)
        return freq, lits, ents

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0]
        subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0]
        obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0]
        return ent_freq, subj_freq, obj_freq


class EntStandardScaler(Scaler):
    """Also scales predicate/relation
    """
    def __init__(
        self, ent_freq, ent_subj, ent_obj, pred_freq, pred_ents, pred_lits
    ) -> None:
        
        # predicate frequency
        self.pred_freq_scaler = StandardScaler()
        input_lst = np.array([int(x) for x in pred_freq.values()]).reshape(-1, 1)
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array([int(x) for x in pred_lits.values()]).reshape(-1, 1)
        self.pred_lits_scaler = StandardScaler()
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array([int(x) for x in pred_ents.values()]).reshape(-1, 1)
        self.pred_ents_scaler = StandardScaler()
        self.pred_ents_scaler.fit(input_lst)

        # Entity Frequency
        input_lst = np.array([int(x) for x in ent_freq.values()]).reshape(-1, 1)
        self.ent_freq_scaler = StandardScaler()
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array([int(x) for x in ent_subj.values()]).reshape(-1, 1)
        self.ent_sub_scaler = StandardScaler()
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array([int(x) for x in ent_obj.values()]).reshape(-1, 1)
        self.ent_obj_scaler = StandardScaler()
        self.ent_obj_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform([[freq]])[0]
        lits = self.pred_lits_scaler.transform([[lits]])[0]
        ents = self.pred_ents_scaler.transform([[ents]])[0]
        return freq, lits, ents

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0]
        subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0]
        obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0]
        return ent_freq, subj_freq, obj_freq


class EntDefaultScaler(Scaler):
    """scales using different sklearn classes"""

    def __init__(
        self,
        ent_freq,
        ent_subj,
        ent_obj,
        pred_freq,
        pred_ents,
        pred_lits,
        scale_class=RobustScaler,
    ) -> None:
        # predicate frequency
        self.pred_freq_scaler = scale_class()
        input_lst = np.array([int(x) for x in pred_freq.values()]).reshape(-1, 1)
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array([int(x) for x in pred_lits.values()]).reshape(-1, 1)
        self.pred_lits_scaler = scale_class()
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array([int(x) for x in pred_ents.values()]).reshape(-1, 1)
        self.pred_ents_scaler = scale_class()
        self.pred_ents_scaler.fit(input_lst)

        # Entity Frequency
        input_lst = np.array([int(x) for x in ent_freq.values()]).reshape(-1, 1)
        self.ent_freq_scaler = scale_class()
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array([int(x) for x in ent_subj.values()]).reshape(-1, 1)
        self.ent_sub_scaler = scale_class()
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array([int(x) for x in ent_obj.values()]).reshape(-1, 1)
        self.ent_obj_scaler = scale_class()
        self.ent_obj_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform([[freq]])[0, 0]
        lits = self.pred_lits_scaler.transform([[lits]])[0, 0]
        ents = self.pred_ents_scaler.transform([[ents]])[0, 0]
        return freq, lits, ents

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0, 0]
        subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0, 0]
        obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0, 0]
        return ent_freq, subj_freq, obj_freq


class EntStandardScalerPredSubjObj(Scaler):
    def __init__(self, pred_obj, pred_subj) -> None:
        # predicate objects for predicates
        self.pred_obj_scaler = StandardScaler()
        input_lst = np.array([int(x) for x in pred_obj.values()]).reshape(-1, 1)
        self.pred_obj_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array([int(x) for x in pred_subj.values()]).reshape(-1, 1)
        self.pred_subj_scaler = StandardScaler()
        self.pred_subj_scaler.fit(input_lst)

    def scale(self, obj, subj):
        subj = self.pred_subj_scaler.transform([[subj]])[0, 0]
        obj = self.pred_obj_scaler.transform([[obj]])[0, 0]
        return obj, subj
