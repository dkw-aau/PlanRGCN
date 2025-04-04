from feature_extraction.sparql_query import Query
import json, os, pandas as pd, configparser
from feature_extraction.constants import PATH_TO_CONFIG
import argparse
import matplotlib.pyplot as plt
from stats.utils import plot_bar_df, plot_bar_specific, bxplot_w_info


class EntityFeatures(Query):
    def __init__(self, endpoint_url):
        super().__init__(endpoint_url)
        self.freq = {}
        self.buckets = None
        self.binner = None

    def get_rdf_entities(self, save_path=None) -> list:
        query_str = f""" SELECT DISTINCT ?e WHERE {{
            {{ ?e ?p ?o .
            FILTER isIRI(?e)}}
            UNION {{?s ?p ?e .
            FILTER isIRI(?e)}}
        }}
        """
        entities = []
        res = self.run_query(query_str)
        for x in res["results"]["bindings"]:
            entities.append(x["e"]["value"])
        if save_path != None:
            with open(save_path, "w") as f:
                json.dump(entities, f)
        return entities

    def get_freq(self, entity: str):
        query_str = f"""
        SELECT  COUNT(*) as ?count WHERE {{
            {{ <{entity}> ?p ?o .
            }}
            UNION {{?s ?p <{entity}> .
            }}
        }}
        """
        res = self.run_query(query_str)
        return res["results"]["bindings"][0]["count"]["value"]

    def get_freq_all_entities(self, lst: list[str], save_path=None):
        for ent in lst:
            self.freq[ent] = self.get_freq(ent)
        if save_path != None:
            with open(save_path, "w") as f:
                json.dump(self.freq, f)
        return self.freq

    def create_bins(self, buckets, verbose=False):
        self.buckets = buckets + 1
        freq_d = {"entity": [], "frequency": []}
        for i in self.freq.keys():
            freq_d["entity"].append(i)
            freq_d["frequency"].append(self.freq[i])
        df = pd.DataFrame.from_dict(freq_d)
        df["frequency"] = df["frequency"].astype("int")
        df = df.sort_values("frequency", ascending=False)
        # df['id'] = [x for x in range(len(df))]
        # df['id'] = df['id'].astype('int')
        # df = df.set_index('id')
        # self.ents = df.iloc[:k]
        # self.ents = self.ents.set_index('entity')
        # self.ents['id'] = [x for x in range(len(self.ents))]
        counts = df["frequency"].drop_duplicates(keep="first")
        counts = pd.DataFrame({"frequency": counts})

        # print(f"Unique Frequencies of entities: {len(counts)}")
        counts["bin"], cut_bin = pd.qcut(
            counts["frequency"], q=buckets, labels=range(buckets), retbins=True
        )
        counts = counts.set_index("frequency")
        df["bin"] = df["frequency"].apply(lambda x: counts.loc[x]["bin"])
        # df['bin'], cut_bin = pd.cut(df['frequency'], bins = bin, labels = range(bin), retbins = True)
        self.intervals = cut_bin
        if verbose:
            print(f"Entity bins: {cut_bin}")
        df = df.set_index("entity")
        self.binner = df
        return df

    def get_feature(self, entity: str):
        try:
            self.binner
        except AttributeError:
            print(f"Cannot get features without get_topk_ents_and_bin being run first")
            exit()
        try:
            bin_no, freq = (
                self.binner.loc[entity]["bin"],
                self.binner.loc[entity]["frequency"],
            )
        except KeyError:
            bin_no, freq = self.buckets, 0
        return bin_no, freq

    def load(parser: configparser.ConfigParser):
        i = EntityFeatures(None)
        entities, freq_dict = load_feature_data(parser)
        i.freq = freq_dict
        return i


def entity_stat_extraction(parser: configparser.ConfigParser):
    global ent_featurizer, entities, freq_dict
    endpoint_url = parser["endpoint"]["endpoint_url"]
    ent_featurizer = EntityFeatures(endpoint_url)
    entity_path = parser["EntityFeaturizer"]["entity_path"]
    entity_freq_dict_path = parser["EntityFeaturizer"]["entity_freq_dict_path"]
    if (not os.path.isfile(entity_path)) or (not os.path.isfile(entity_freq_dict_path)):
        entities = ent_featurizer.get_rdf_entities(save_path=entity_path)
        freq_dict = ent_featurizer.get_freq_all_entities(
            entities, save_path=entity_freq_dict_path
        )
    else:
        entities = json.load(open(entity_path, "r"))
        freq_dict = json.load(open(entity_freq_dict_path, "r"))
        ent_featurizer.freq = freq_dict
    print(f"Number of entities: {len(entities)}")


def load_feature_data(parser: configparser.ConfigParser):
    entity_path = parser["EntityFeaturizer"]["entity_path"]
    entity_freq_dict_path = parser["EntityFeaturizer"]["entity_freq_dict_path"]
    entities = json.load(open(entity_path, "r"))
    freq_dict = json.load(open(entity_freq_dict_path, "r"))
    return entities, freq_dict


def get_frequency_df(freq_dict):
    freq = {"entity": [], "frequency": []}
    for k in freq_dict.keys():
        freq["entity"].append(k)
        freq["frequency"].append(freq_dict[k])
    df = pd.DataFrame.from_dict(freq)
    df["frequency"] = df["frequency"].astype("int")
    df = df.sort_values("frequency", ascending=False)
    return df


def plot_entity_stat(entities, freq_dict, save=None, option="bxp"):
    plt.clf()
    df = get_frequency_df(freq_dict)
    assert len(df) == len(entities)
    # plot_bar_specific(freq_dict)
    match option:
        case "bar":
            df = df[:30]
            plot_bar_df(df, "frequency", "entity", save=save)
        case "bxp":
            bxp_dat = get_bxp_data(df)
            bxplot_w_info(bxp_dat, "Entity Count", save=save)
            pass

    # ax = df.plot(kind='bar',x='entity', y='frequency')
    # ax.bar_label(ax.containers[0],label_type='edge')
    # ax.set_ylabel("Entity Frequency Distribution over RDF graph")
    # plt.yscale('log')
    # if save == None:
    #    plt.show()
    # else:
    #    plt.savefig(save)


def get_bxp_data(df: pd.DataFrame):
    df = df[~(df["frequency"].isin([1, 2, 3, 4]))]
    data = [
        {
            "label": "Entity Distribution",
            "whislo": df["frequency"].min(),  # Bottom whisker position
            "q1": df["frequency"].quantile(q=0.25),  # First quartile (25th percentile)
            "med": df["frequency"].quantile(q=0.5),  # Median         (50th percentile)
            "q3": df["frequency"].quantile(q=0.75),  # Third quartile (75th percentile)
            "whishi": df["frequency"].max(),  # Top whisker position
            "fliers": [],  # Outliers
        }
    ]
    print(data)
    return data


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(prog="PredicateFeaturizer_subj_obj)")
    arg_parse.add_argument("cmd")
    args = arg_parse.parse_args()

    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    match (args.cmd):
        case "run":
            entity_stat_extraction(parser)
        case "plot":
            entities, freq_dict = load_feature_data(parser)
            plot_entity_stat(
                entities,
                freq_dict,
                save=parser["Results"]["fig_dir"] + "entity_dist.png",
            )
            pass
        case "binnify":
            featurizer = EntityFeatures.load(parser)
            keys = list(featurizer.freq.keys())
            keys = [x for x in keys if not ".openlinksw" in x]
            print(keys[:10])
            featurizer.create_bins(buckets=20)
            print(featurizer.get_feature(keys[-1]))

        case other:
            print("Please choose a valid option")
