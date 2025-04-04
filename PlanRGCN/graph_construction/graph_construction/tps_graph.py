import dgl
import json
from feature_extraction.base_featurizer import BaseFeaturizer
from graph_construction.bgp import BGP
import networkx as nx

from graph_construction.nodes.node import Node
from graph_construction.triple_pattern import TriplePattern


class tps_graph:
    """Class representing a execution of a BGP"""

    def __init__(self, bgp: BGP, featurizer) -> None:
        assert isinstance(featurizer, BaseFeaturizer)
        self.bgp = bgp
        for i, tp in enumerate(self.bgp.triples):
            tp.id = i
        self.featurizer = featurizer
        self.nodes: list[Node] = []
        self.edges: list[(int, int, int)] = self.get_edges(
            self.bgp.triples
        )  # triple1 triple2 relationtype
        self.graph = nx.DiGraph()
        self.graph_creation()

    def graph_creation(self):
        vals = {}
        self.graph: nx.DiGraph
        for tp1_idx, tp2_idx, edge_type in self.edges:
            # self.graph.add_edge(self.bgp.triples[tp1_idx],self.bgp.triples[tp2_idx], rel_type=edge_type)
            self.graph.add_edge(tp1_idx, tp2_idx, rel_type=edge_type)
            if not tp1_idx in vals.keys():
                vals[tp1_idx] = self.featurizer.get_feat_vec(self.bgp.triples[tp1_idx])
            if not tp2_idx in vals.keys():
                vals[tp2_idx] = self.featurizer.get_feat_vec(self.bgp.triples[tp2_idx])
        nx.set_node_attributes(self.graph, vals, "node_features")

    def get_edges(self, triples: list[TriplePattern]):
        edges = []
        c_vars_lst = []
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                c_vars = self.get_common_variable(triples[i], triples[j])
                c_vars_lst.append(len(c_vars))
                if len(c_vars) == 0:
                    continue
                local_edges = self.get_common_edge_type(triples[i], triples[j], c_vars)
                edges.extend(local_edges)
        assert len(edges) > 0
        return edges

    def get_common_variable(self, trp1: TriplePattern, trp2: TriplePattern):
        trp1_vars = set(trp1.get_joins())
        trp2_vars = set(trp2.get_joins())
        c_vars = trp1_vars.intersection(trp2_vars)
        return list(c_vars)

    def get_common_edge_type(
        self, trp1: TriplePattern, trp2: TriplePattern, common_vars
    ):
        edges = []
        for c in common_vars:
            edges.append((trp1.id, trp2.id, self.get_relation_type(trp1, trp2, c)))
        return edges

    def get_relation_type(
        self, trp1: TriplePattern, trp2: TriplePattern, common_variable
    ):
        if trp1.subject == common_variable and trp2.subject == common_variable:
            return 0
        if trp1.subject == common_variable and trp2.predicate == common_variable:
            return 1
        if trp1.subject == common_variable and trp2.object == common_variable:
            return 2

        if trp1.predicate == common_variable and trp2.subject == common_variable:
            return 3
        if trp1.predicate == common_variable and trp2.predicate == common_variable:
            return 4
        if trp1.predicate == common_variable and trp2.object == common_variable:
            return 5
        if trp1.object == common_variable and trp2.subject == common_variable:
            return 6
        if trp1.object == common_variable and trp2.predicate == common_variable:
            return 7
        if trp1.object == common_variable and trp2.object == common_variable:
            return 8
        """if trp1.object == common_variable and trp2.subject == common_variable:
            return 6
        if trp1.object == common_variable and trp2.predicate == common_variable:
            return 7
        if trp1.object == common_variable and trp2.object == common_variable:
            return 8"""


# we treat joins on constants as different relation
class tps_cons_graph(tps_graph):
    """Triple pattern graph where const joins are treated as the same as variable joins"""

    def __init__(self, bgp: BGP, featurizer) -> None:
        super().__init__(bgp, featurizer)

    def get_relation_type(
        self, trp1: TriplePattern, trp2: TriplePattern, common_variable
    ):
        if (
            trp1.subject == common_variable
            and trp2.subject == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 0
        if (
            trp1.subject == common_variable
            and trp2.predicate == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 1
        if (
            trp1.subject == common_variable
            and trp2.object == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 2

        if (
            trp1.predicate == common_variable
            and trp2.subject == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 3
        if (
            trp1.predicate == common_variable
            and trp2.predicate == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 4
        if (
            trp1.predicate == common_variable
            and trp2.object == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 5
        if (
            trp1.object == common_variable
            and trp2.subject == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 6
        if (
            trp1.object == common_variable
            and trp2.predicate == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 7
        if (
            trp1.object == common_variable
            and trp2.object == common_variable
            and trp1.subject.type == "VAR"
        ):
            return 8

        # const joins
        if (
            trp1.subject == common_variable
            and trp2.subject == common_variable
            and trp1.subject.type == "URI"
        ):
            return 9
        if (
            trp1.subject == common_variable
            and trp2.predicate == common_variable
            and trp1.subject.type == "URI"
        ):
            return 10
        if (
            trp1.subject == common_variable
            and trp2.object == common_variable
            and trp1.subject.type == "URI"
        ):
            return 11

        if (
            trp1.predicate == common_variable
            and trp2.subject == common_variable
            and trp1.subject.type == "URI"
        ):
            return 12
        if (
            trp1.predicate == common_variable
            and trp2.predicate == common_variable
            and trp1.subject.type == "URI"
        ):
            return 13
        if (
            trp1.predicate == common_variable
            and trp2.object == common_variable
            and trp1.subject.type == "URI"
        ):
            return 14
        if (
            trp1.object == common_variable
            and trp2.subject == common_variable
            and trp1.subject.type == "URI"
        ):
            return 15
        if (
            trp1.object == common_variable
            and trp2.predicate == common_variable
            and trp1.subject.type == "URI"
        ):
            return 16
        if (
            trp1.object == common_variable
            and trp2.object == common_variable
            and trp1.subject.type == "URI"
        ):
            return 17


def get_tp_graph_class(t):
    assert isinstance(t, str)
    if t == "tp":
        return tps_graph
    elif t == "tp_const":
        return tps_cons_graph


def create_dummy_dgl_graph():
    bpgs_string = '{"[?x http://www.wikidata.org/prop/direct/P1936 ?z, ?y http://www.wikidata.org/prop/direct/P1652 ?x]": {"with_runtime": 421605504, "without_runtime": 290386128, "with_size": 0, "without_size": 0}}'
    bgp_dict = json.loads(bpgs_string)
    bgp_string = list(bgp_dict.keys())[0]
    info = bgp_dict[bgp_string]
    bgp = BGP(bgp_string, info)
    train_log = "/work/data/confs/May2/debug_train.json"
    featurizer = BaseFeaturizer(
        train_log=train_log,
        is_pred_clust_feat=True,
        save_pred_graph_png=None,
        community_no=10,
        path_pred_clust={
            "save_path": None,
            "load_path": "/work/data/confs/May2/pred_clust.json",
        },
    )

    tps = tps_graph(bgp, featurizer=featurizer)
    dgl_graph = dgl.from_networkx(
        tps.graph, edge_attrs=["rel_type"], node_attrs=["node_features"]
    )
    dgl_graph = dgl.add_self_loop(dgl_graph)
    return dgl_graph


if __name__ == "__main__":
    # test
    bpgs_string = '{"[?x http://www.wikidata.org/prop/direct/P1936 ?z, ?y http://www.wikidata.org/prop/direct/P1652 ?x]": {"with_runtime": 421605504, "without_runtime": 290386128, "with_size": 0, "without_size": 0}}'
    bgp_dict = json.loads(bpgs_string)
    bgp_string = list(bgp_dict.keys())[0]
    info = bgp_dict[bgp_string]
    bgp = BGP(bgp_string, info)
    train_log = "/work/data/confs/May2/debug_train.json"
    featurizer = BaseFeaturizer(
        train_log=train_log,
        is_pred_clust_feat=True,
        save_pred_graph_png=None,
        community_no=10,
        path_pred_clust={
            "save_path": None,
            "load_path": "/work/data/confs/May2/pred_clust.json",
        },
    )

    tps = tps_graph(bgp, featurizer=featurizer)
    dgl_graph = dgl.from_networkx(
        tps.graph, edge_attrs=["rel_type"], node_attrs=["node_features"]
    )
    print(dgl_graph.edata)
    print(dgl_graph.ndata)
    # for (i,v) in tps.graph.edges:
    #    print(tps.graph.edges[i,v]['rel_type'])
