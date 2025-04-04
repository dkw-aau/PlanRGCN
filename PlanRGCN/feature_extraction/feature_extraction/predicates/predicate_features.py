from feature_extraction.sparql_query import Query
import json
import pickle as pcl
import time, pandas as pd, numpy as np
import traceback
from datetime import datetime
from feature_extraction.constants import PATH_TO_CONFIG
import configparser
import os

# Class definitions for extracting predicate features
import copy
from SPARQLWrapper import SPARQLWrapper, JSON, POST
import pathlib


class PredicateFeaturesQuery(Query):
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url)
        self.start = time.time()
        self.uniqueLiteralCounter = {}
        self.unique_entities_counter = {}
        self.predicate_freq = {}
        if hasattr(self, "sparql") and timeout != None:
            self.sparql.setTimeout(timeout)

    #
    def extract_predicate_features(
        self,
        predicates=None,
        save_decode_err_preds="/work/data/decode_error_pred.json",
        save_path=None,
    ):
        if predicates == None:
            predicates = self.predicates
        decode_errors = []
        for pred_no, pred in enumerate(predicates):
            try:
                self.set_query_unique_literal_predicate_v2(pred, number=pred_no)
            except (json.decoder.JSONDecodeError, Exception, TimeoutError):
                traceback.print_exc()
                decode_errors.append(pred)
            try:
                self.set_query_unique_entity_predicate(pred, number=pred_no)
            except (json.decoder.JSONDecodeError, Exception, TimeoutError):
                traceback.print_exc()
                decode_errors.append(pred)
            try:
                self.set_predicate_freq(pred, number=pred_no)
            except (json.decoder.JSONDecodeError, Exception, TimeoutError):
                traceback.print_exc()
                decode_errors.append(pred)
            if save_path != None and (pred_no % 50 == 1):
                self.save(save_path)
        json.dump(predicates, open(save_decode_err_preds, "w"))

    #

    # second pass over dataset
    # call load first
    def extract_features_for_remaining(
        self,
        predicates=None,
        save_decode_err_preds=f'/work/data/confs/newPredExtractionRun/decode_error_pred_additional_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.json',
        save_path=None,
    ):
        if predicates == None:
            predicates = self.predicates
        decode_errors = []
        for pred_no, pred in enumerate(predicates):
            if not pred in self.uniqueLiteralCounter.keys():
                try:
                    self.set_query_unique_literal_predicate_v2(pred, number=pred_no)
                except (json.decoder.JSONDecodeError, Exception, TimeoutError):
                    # traceback.print_exc()
                    decode_errors.append(pred)
            if not pred in self.unique_entities_counter.keys():
                try:
                    self.set_query_unique_entity_predicate(pred, number=pred_no)
                except (json.decoder.JSONDecodeError, Exception, TimeoutError):
                    # traceback.print_exc()
                    decode_errors.append(pred)
            if not pred in self.predicate_freq.keys():
                try:
                    self.set_predicate_freq(pred, number=pred_no)
                except (json.decoder.JSONDecodeError, Exception, TimeoutError):
                    # traceback.print_exc()
                    decode_errors.append(pred)
            if save_path != None and (pred_no % 50 == 1):
                self.save(save_path)
        if save_path != None:
            self.save(save_path)
        json.dump(predicates, open(save_decode_err_preds, "w"))

    def set_query_unique_literal_predicate(self, predicate, number=None):
        query_str = f"""SELECT ?s ?o WHERE {{
            ?s <{predicate}> ?o .
        }}
        """
        res = self.run_query(query_str)
        self.process_freq_features(res, predicate, number=number)

    #
    def set_query_unique_literal_predicate_v2(self, predicate, number=None):
        query_str = f"""SELECT (COUNT(DISTINCT ?o) AS ?literals) WHERE {{
            ?s <{predicate}> ?o .
            FILTER(isLiteral(?o))
        }}
        """
        val = None
        try:
            res = self.run_query(query_str)
            self.uniqueLiteralCounter[predicate] = res["results"]["bindings"][0][
                "literals"
            ]["value"]
            val = res["results"]["bindings"][0]["literals"]["value"]
            print(
                f"{time.time()-self.start:.2f}{number}{predicate}: {res['results']['bindings'][0]['literals']['value']}"
            )
            return val
        except RuntimeError or Exception or TimeoutError:
            self.uniqueLiteralCounter[predicate] = -1
            return None
        # self.process_freq_features(res, predicate, number=number)

    #
    def set_query_unique_entity_predicate(self, predicate, number=None):
        query_str = f"""SELECT (COUNT(DISTINCT ?e) AS ?entities) WHERE {{
            {{?e <{predicate}> ?o .
            FILTER(isURI(?e))}}
            UNION {{?s <{predicate}> ?e .
            FILTER(isURI(?e))}}
        }}
        """
        try:
            res = self.run_query(query_str)
            print(
                f"ENT {predicate}: {res['results']['bindings'][0]['entities']['value']}"
            )
            self.unique_entities_counter[predicate] = res["results"]["bindings"][0][
                "entities"
            ]["value"]

        except RuntimeError or Exception or TimeoutError as e:
            print(e.pri)
            self.unique_entities_counter[predicate] = -1

    #
    def set_predicate_freq(self, predicate, number=None):
        query_str = f"""SELECT (COUNT(*) AS ?triples) WHERE {{
            ?s <{predicate}> ?o .
        }}
        """
        val = None
        try:
            res = self.run_query(query_str)
            self.predicate_freq[predicate] = res["results"]["bindings"][0]["triples"][
                "value"
            ]
            val = res["results"]["bindings"][0]["triples"]["value"]
            print(
                f"FREQ {predicate}: {res['results']['bindings'][0]['triples']['value']}"
            )
            return val
        except RuntimeError or Exception or TimeoutError:
            self.predicate_freq[predicate] = -1
            return None

    #
    def process_freq_features(
        self, sparql_result, predicate, verbose=True, number=None
    ):
        unique_literals = set()
        unique_ents = set()
        pattern_count = 0
        print(
            f"Total binding for {predicate}: {len(sparql_result['results']['bindings'])}"
        )
        for x in sparql_result["results"]["bindings"]:
            pattern_count += 1
            if x["o"]["type"] == "literal":
                unique_literals.add(x["o"]["value"])
            elif x["o"]["type"] == "uri":
                unique_ents.add(x["o"]["value"])
            if x["s"]["type"] == "uri":
                unique_ents.add(x["s"]["value"])
        self.uniqueLiteralCounter[predicate] = len(unique_literals)
        self.unique_entities_counter[predicate] = len(unique_ents)
        self.predicate_freq[predicate] = pattern_count
        if verbose:
            self.print_predicat_stat(
                predicate,
                len(unique_literals),
                len(unique_ents),
                pattern_count,
                number=number,
            )

    #
    def save(self, path):
        with open(path, "wb") as f:
            pcl.dump(self, f)

    #
    def print_predicat_stat(
        self,
        predicate: str,
        unique_literals: int,
        unique_ents: int,
        pattern_count: int,
        number=None,
    ):
        if number == None:
            number = ""
        else:
            number = f" : {number:05}"
        print(f"{time.time()-self.start:.2f}{number} Stats for {predicate}")
        print(f"\t# of triples: {pattern_count}")
        print(f"\t# of unique entities: {unique_ents}")
        print(f"\t# of unique literals: {unique_literals}")

    #
    def get_rdf_predicates(self, save_path=None) -> list:
        query_str = f""" SELECT DISTINCT ?p WHERE {{
            ?s ?p ?o
        }}
        """
        predicates = []
        res = self.run_query(query_str)
        for x in res["results"]["bindings"]:
            predicates.append(x["p"]["value"])
        if save_path != None:
            p = pathlib.Path(save_path)
            pathlib.Path(str(p.parent)).mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(predicates, f)
        return predicates

    #
    def load_predicates(self, path):
        self.predicates = json.loads(open(path, "r").read())
        return self.predicates

    #
    def prepare_pred_feat(self, bins=30, k=20):
        dct = {"predicate": [], "freq": []}
        for key in self.predicate_freq.keys():
            dct["predicate"].append(key)
            dct["freq"].append(self.predicate_freq[key])
        df = pd.DataFrame.from_dict(dct)
        # df_freq = df['freq'].astype('int')
        df["freq"] = pd.to_numeric(df["freq"])
        df_freq = df.sort_values("freq", ascending=False)
        df = df_freq
        df_freq = df_freq.assign(row_number=range(len(df_freq)))
        df_freq = df_freq.set_index("row_number")
        # print(df_freq)
        # print(df_freq.columns)
        # print(df_freq.index)
        df_freq = df_freq.loc[:k]
        df_freq = df_freq.reset_index().set_index("predicate")

        # unique value binning
        # _, cut_bin= pd.qcut(df['freq'], q = bins, retbins = True, duplicates='drop')
        # df['bin'], cut_bin = pd.qcut(df['freq'], q = bins, labels = [x for x in range(len(cut_bin)-1)], retbins = True, duplicates='drop')
        df["bin"], cut_bin = pd.qcut(
            df["freq"], q=bins, labels=[x for x in range(bins)], retbins=True
        )
        # max_bin = df['bin'].max()
        df = df.set_index("predicate")
        self.predicate_bin_df = df
        self.bin_vals = cut_bin
        # self.pretty_print_buckets()
        self.freq_k = k
        self.total_bin = len(cut_bin) + 1

        self.topk_df = df_freq

    def pretty_print_buckets(self):
        print(f"Used bucket intervas: ")
        for i in range(0, len(self.bin_vals) - 1):
            print(
                f"{i+1} [{round(self.bin_vals[i])}, {round(self.bin_vals[i+1])}]",
                end="\n",
            )
        print("\n")

    def top_k_predicate(self, predicate):
        try:
            return self.topk_df.loc[predicate]["row_number"]
        except KeyError:
            return None

    #
    def get_bin(self, predicate):
        try:
            return self.predicate_bin_df.loc[predicate]["bin"]
        except KeyError as e:
            # print(predicate)
            return self.total_bin + 1

    #
    # deprecated
    def binnify(self, predicate_freq):
        bin_counter = 0
        for x in range(1, len(self.bin_vals)):
            if (
                self.bin_vals[x - 1] < predicate_freq
                and predicate_freq < self.bin_vals[x]
            ):
                return bin_counter
            bin_counter += 1
        return bin_counter

    def convert_dict_vals_to_int(self, dct: dict):
        for k in dct.keys():
            dct[k] = int(dct[k])
        return dct

    # Used to load existing object
    def load(path):
        obj = load_pickle(path)
        if hasattr(obj, "endpoint_url"):
            endpoint_url = obj.endpoint_url
        else:
            endpoint_url = None
        i = PredicateFeaturesQuery(endpoint_url)
        i.uniqueLiteralCounter = i.convert_dict_vals_to_int(obj.uniqueLiteralCounter)
        i.predicate_freq = i.convert_dict_vals_to_int(obj.predicate_freq)
        i.unique_entities_counter = i.convert_dict_vals_to_int(
            obj.unique_entities_counter
        )

        return i

    def prepare_pred_featues_for_bgp(path, bins=30, topk=15):
        i = PredicateFeaturesQuery.load(path)
        i.prepare_pred_feat(bins=bins, k=topk)
        return i


def iterate_results(sparql_results):
    for x in sparql_results["results"]["bindings"]:
        for v in sparql_results["head"]["vars"]:
            print(v, x[v])


def print_bindings_stats(sparql_results):
    binding_count = 0
    literals = []
    for x in sparql_results["results"]["bindings"]:
        binding_count += 1
    print(f"number of bindings are {binding_count}")


# Functions for extracting statistics:


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pcl.load(f)
        return obj
    return None


# Procedures for extracting differrent statistics.


def test():
    q = Query("http://dbpedia.org/sparql")
    q.run_query("SELECT * WHERE { ?s ?p ?o} LIMIT 1")
    print(q.results.keys())
    iterate_results(q.results)
    # for x in q.results['results']['bindings']:
    #    for v in q.results['head']['vars']:
    #        print(v,x[v])


def extractPredicateFeatures(parser: configparser.ConfigParser):
    # q = PredicateFeaturesQuery("http://172.21.232.208:3030/jena/sparql")
    q = PredicateFeaturesQuery(parser["endpoint"]["endpoint_url"])
    if os.path.isfile(parser["PredicateFeaturizer"]["predicate_path"]):
        preds = q.load_predicates(parser["PredicateFeaturizer"]["predicate_path"])
    else:
        preds = q.get_rdf_predicates(
            save_path=parser["PredicateFeaturizer"]["predicate_path"]
        )

    print(f"# preds {len(preds)}")
    q.extract_predicate_features(
        preds,
        save_path=parser["PredicateFeaturizer"]["predicate_featurizer_path"],
        save_decode_err_preds=parser["PredicateFeaturizer"]["failure_path"],
    )
    q.save(parser["PredicateFeaturizer"]["predicate_featurizer_path"])
    # q.set_query_unique_literal_predicate("<http://www.wikidata.org/prop/direct/P5395>")
    # iterate_results(q.results)
    # print_bindings_stats(q.results)


def run_continued():
    # q = PredicateFeaturesQuery("https://query.wikidata.org/sparql")
    # q = PredicateFeaturesQuery("http://172.21.232.208:3030/jena/sparql")
    # preds = q.get_rdf_predicates(save_path='/work/data/predicates.json')
    # preds = q.load_predicates('/work/data/specific_graph/predicates.json')
    # print(f"# preds {len(preds)}")
    with open("/work/data/specific_graph/pred_feat.pickle", "rb") as f:
        q = pcl.load(f)
    return q
    # q.extract_predicate_features(preds, save_path='/work/data/specific_graph/pred_feat.pickle')
    # q.save('/work/data/pred_feat.pickle')
    # q.set_query_unique_literal_predicate("<http://www.wikidata.org/prop/direct/P5395>")
    # iterate_results(q.results)
    # print_bindings_stats(q.results)


def run_last_predicates():
    predicates = json.load(
        open("/work/data/confs/newPredExtractionRun/predicates_only.json", "r")
    )
    print(f"Number of total predicates: {len(predicates)}")
    path_featurizer = "/work/data/pred_feat.pickle"
    endpoint_url = "http://172.21.233.23:8891/sparql/"
    save_path = f'/work/data/confs/newPredExtractionRun/pred_feat_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.pickle'
    # Prepare predicate Featurizer
    featurizer = PredicateFeaturesQuery.load(path_featurizer)
    featurizer.sparql = SPARQLWrapper(endpoint_url)
    featurizer.sparql.setReturnFormat(JSON)
    featurizer.sparql.setMethod(POST)
    featurizer.start = time.time()

    featurizer.extract_features_for_remaining(
        predicates=predicates, save_path=save_path
    )


def get_value_array(dct: dict):
    keys = list(dct.keys())
    freqs = []
    no_feat = []
    for x in keys:
        if dct[x] == "-1":
            no_feat.append(x)
        freqs.append(int(dct[x]))
    return freqs, no_feat, len(keys)


def extract_meta_pred_freq_stats(pred_feats):
    freqs, no_feat, els = get_value_array(pred_feats.predicate_freq)
    freqs = np.array(freqs)
    return (
        np.min(freqs),
        np.max(freqs),
        int(np.quantile(freqs, q=0.25)),
        int(np.quantile(freqs, q=0.5)),
        int(np.quantile(freqs, q=0.75)),
        no_feat,
        els,
    )


def extract_meta_pred_literal_stats(pred_feats):
    freqs, no_feat, els = get_value_array(pred_feats.uniqueLiteralCounter)
    freqs = np.array(freqs)
    return (
        np.min(freqs),
        np.max(freqs),
        int(np.quantile(freqs, q=0.25)),
        int(np.quantile(freqs, q=0.5)),
        int(np.quantile(freqs, q=0.75)),
        no_feat,
        els,
    )


def extract_meta_pred_entity_stats(pred_feats):
    freqs, no_feat, els = get_value_array(pred_feats.unique_entities_counter)
    freqs = np.array(freqs)
    return (
        np.min(freqs),
        np.max(freqs),
        int(np.quantile(freqs, q=0.25)),
        int(np.quantile(freqs, q=0.5)),
        int(np.quantile(freqs, q=0.75)),
        no_feat,
        els,
    )


def check_stats(path="/work/data/pred_feat.pickle"):
    pred_features = PredicateFeaturesQuery.load(path)
    # pred_features = load_pickle('/work/data/pred_feat.pickle')
    print("Pred Features succesfully extracted")

    for function, f_name in zip(
        [
            extract_meta_pred_freq_stats,
            extract_meta_pred_literal_stats,
            extract_meta_pred_entity_stats,
        ],
        ["Predicate Frequencies", "Literals", "Entities"],
    ):
        freq_min, freq_max, q_25, q_50, q_75, no_feat, els = function(pred_features)
        print(f"Meta statistics of {f_name}")
        print(
            f"\t Min: {freq_min:10d}\n\t 25%: {q_25:10d}\n\t 50%: {q_50:10d}\n\t 75%: {q_75:10d}\n\t Max: {freq_max:10d}\n\tkeys: {els:10d}"
        )
        print("\n\n")

    predicates = list(pred_features.predicate_freq.keys())
    pred_features.prepare_pred_feat()
    print(f"binned predicate count: {len(pred_features.predicate_bin_df.index)}")


if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    extractPredicateFeatures()
    check_stats(path=parser["PredicateFeaturizer"]["predicate_featurizer_path"])
    # check_stats(path='/work/data/confs/newPredExtractionRun/pred_feat_01_04_2023_07_48.json')

    # run_last_predicates()
    # q =run_continued()
