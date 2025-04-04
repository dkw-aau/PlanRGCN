from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

class BinnerEntPred:
    def __init__(
        self,
        ent_freq,
        ent_subj,
        ent_obj,
        pred_freq,
        pred_ents,
        pred_lits,
        lits,
        bins=50,
        random_state=42,
    ) -> None:
        # predicate frequency

        # KBinsDiscretizer(n_bins=50, strategy="quantile", encode="ordinal")
        self.pred_freq_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        input_lst = np.array(sorted([int(x) for x in pred_freq.values()])).reshape(
            -1, 1
        )
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array(sorted([int(x) for x in pred_lits.values()])).reshape(
            -1, 1
        )
        self.pred_lits_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array(sorted([int(x) for x in pred_ents.values()])).reshape(
            -1, 1
        )
        self.pred_ents_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.pred_ents_scaler.fit(input_lst)
        # Entity Frequency
        input_lst = np.array(sorted([int(x) for x in ent_freq.values()])).reshape(-1, 1)
        self.ent_freq_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array(sorted([int(x) for x in ent_subj.values()])).reshape(-1, 1)
        self.ent_sub_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array(sorted([int(x) for x in ent_obj.values()])).reshape(-1, 1)
        self.ent_obj_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.ent_obj_scaler.fit(input_lst)
        
        # Literal Frequency
        if lits is not None:
            input_lst = np.array(sorted([int(x) for x in lits.values()])).reshape(-1, 1)
            self.lit_freq_scaler = KBinsDiscretizer(
                n_bins=bins,
                strategy="quantile",
                encode="onehot-dense",
                random_state=random_state,
            )
            self.lit_freq_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform([[freq]])[0]
        lits = self.pred_lits_scaler.transform([[lits]])[0]
        ents = self.pred_ents_scaler.transform([[ents]])[0]
        # return np.log(freq), np.log(lits), np.log(ents)
        return freq, lits, ents

    def pred_scale_len(self):
        return (
            self.pred_freq_scaler.n_bins_[0]
            + self.pred_lits_scaler.n_bins_[0]
            + self.pred_ents_scaler.n_bins_[0]
        )

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0]
        subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0]
        obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0]
        return ent_freq, subj_freq, obj_freq

    def ent_scale_len(self):
        return (
            self.ent_freq_scaler.n_bins_[0]
            + self.ent_sub_scaler.n_bins_[0]
            + self.ent_obj_scaler.n_bins_[0]
        )

    def ent_scale_no_values(self):
        ent_freq = np.zeros(self.ent_freq_scaler.n_bins_[0])
        subj_freq = np.zeros(self.ent_sub_scaler.n_bins_[0])
        obj_freq = np.zeros(self.ent_obj_scaler.n_bins_[0])
        return ent_freq, subj_freq, obj_freq
    
    def lit_scale(self, lit_freq):
        ent_freq = self.lit_freq_scaler.transform([[lit_freq]])[0]
        return ent_freq

    def lit_scale_len(self):
        return (
            self.lit_freq_scaler.n_bins_[0]
        )

    def lit_scale_no_values(self):
        ent_freq = np.zeros(self.lit_freq_scaler.n_bins_[0])
        return ent_freq
