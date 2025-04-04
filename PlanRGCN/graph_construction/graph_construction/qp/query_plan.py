import dgl
from graph_construction.qp.qp_utils import QueryPlanUtils, get_relation_types
import networkx as nx
import numpy as np
from graph_construction.stack import Stack
from graph_construction.feats.featurizer import FeaturizerBase, FeaturizerPredStats
from graph_construction.node import Node, FilterNode, TriplePattern
from graph_construction.nodes.path_node import PathNode
import torch as th


class QueryPlan:
    max_relations = 16

    def __init__(self, data) -> None:
        QueryPlan.set_max_rels(number=16)
        self.process(data)
        
    def set_max_rels(number=16):
        QueryPlan.max_relations = number
    
    def process(self, data):
        self.level = 0
        self.data = data
        self.triples: list[TriplePattern] = list()
        self.filters: list[FilterNode] = list()
        self.edges = list()

        self.join_vars = {}
        self.filter_dct = {}
        self.op_lvl = {}

        self.iterate_ops(self.add_triple, "Triple")
        self.iterate_ops(self.add_filters, "filter")
        # self.add_self_loop_triples()
        self.assign_trpl_ids()
        self.assign_filt_ids()

        # filt_edge = [(d.id, b.id, r) for (d, b, r) in self.edges if r == 9]
        # print(filt_edge)
        # self.iterate_ops(self.add_binaryOP, "minus")
        # self.iterate_ops(self.add_binaryOP, "union")
        self.iterate_ops(self.add_binaryOP, "join")
        self.iterate_ops(self.add_binaryOP, "leftjoin")
        self.iterate_ops(self.add_binaryOP, "conditional")
        self.add_sngl_trp_rel()
        # can be used to check for existence of operators
        # self.iterate_ops(self.assert_operator, "leftjoin")
        # self.iterate_ops(self.assert_operator, "join")
        # self.iterate_ops(self.assert_operator, "diff")
        # self.iterate_ops(self.assert_operator, "lateral")

        # print(self.edges)
        self.nodes = [x.id for x in self.triples]
        self.nodes.extend([x.id for x in self.filters])

        self.node2obj = {}
        self.initialize_node2_obj()
        self.G = self.networkx()
        self.G.add_nodes_from(self.nodes)

    def initialize_node2_obj(self):
        for t in self.triples:
            if t.id in self.node2obj.keys():
                raise Exception("Existing node createad")
            self.node2obj[t.id] = t
        for f in self.filters:
            if f.id in self.node2obj.keys():
                raise Exception("Existing node createad")
            self.node2obj[f.id] = f

    def iterate_ops(self, func, node_type: str):
        current = self.data
        current["level"] = self.level
        if current == None:
            return
        stack = Stack()
        stack.push(current)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current:
                if current["opName"] == "BGP":
                    self.iterate_bgp(current, func, node_type, filter=None)
                else:
                    self.level += 1
                    for node in reversed(current["subOp"]):
                        node["level"] = self.level
                        stack.push(node)
            if current["opName"] == node_type:
                func(current)
            #if current["opName"] == "join":
            #    raise Exception("We have a join here")

    def iterate_bgp(self, data, func, node_type, filter=None):
        self.level += 1
        for triple in data["subOp"]:
            triple["level"] = self.level
            if triple["opName"] == node_type:
                func(triple, add_data=filter)

    def add_self_loop_triples(self):
        for t in self.triples:
            for v in t.get_joins():
                self.edges.append((t, t, self.get_join_type(t, t, v)))

    def add_triple(self, data, add_data=None):
        t = TriplePattern(data)
        join_v = t.get_joins()
        for v in join_v:
            if v in self.join_vars.keys():
                triple_lst = self.join_vars[v]
                for tp in triple_lst:
                    self.edges.append((tp, t, self.get_join_type(tp, t, v)))
                self.join_vars[v].append(t)
            else:
                self.join_vars[v] = [t]
        self.triples.append(t)
        """if add_data != None:
            add_data: FilterNode
            for v in add_data.vars:
                t_var_labels = [tv.node_label for tv in t.get_joins()]
                if v in t_var_labels:
                    print(t)"""

    def add_filters(self, data, add_data=None):
        self.filters
        self.join_vars
        filter_node = FilterNode(data)
        self.filters.append(filter_node)
        filter_triples = QueryPlanUtils.extract_triples(data)
        expr_string = data["expr"]
        expr_string = expr_string.replace(")", " ")
        filt_vars = [
            x
            for x in expr_string.split(" ")
            if (x.startswith("?") or x.startswith("$"))
        ]
        filter_triples = QueryPlanUtils.map_extracted_triples(
            filter_triples, self.triples
        )
        filt_vars = list(set(filt_vars))

        # dobbel check with filter_node.vars field insead of filt_vars
        for v in filt_vars:
            if v in self.join_vars.keys():
                for t in filter_triples:
                    # added to check whether filter is concerned with triple pattersn
                    if v in t.get_variables():
                        self.edges.append(
                            (t, filter_node, self.get_join_type(t, filter_node, v))
                        )

    def get_join_type(self, trp1, trp2, common_variable):
        return get_relation_types(trp1, trp2, common_variable)

    def assign_trpl_ids(self):
        self.max_id = 0
        for i, t in enumerate(self.triples):
            t.id = i
            self.max_id = np.max([t.id, self.max_id])

    def assign_filt_ids(self):
        for i, t in enumerate(self.filters):
            n_id = 1 + i + self.max_id
            t.id = n_id
            self.max_id = np.max([n_id, self.max_id])

    def add_binaryOP(self, data, add_data=None):
        assert len(data["subOp"]) == 2
        left = data["subOp"][0]
        right = data["subOp"][1]
        left_triples = QueryPlanUtils.extract_triples(left)
        left_triples = QueryPlanUtils.map_extracted_triples(left_triples, self.triples)
        right_triples = QueryPlanUtils.extract_triples(right)
        right_triples = QueryPlanUtils.map_extracted_triples(
            right_triples, self.triples
        )
        for r in right_triples:
            r: TriplePattern
            for l in left_triples:
                l_vars = l.get_joins()
                for r_v in r.get_joins():
                    if r_v in l_vars:
                        # consider adding the other way for union as a special case
                        self.edges.append(
                            (l, r, QueryPlanUtils.get_relations(data["opName"]))
                        )
        # print(left_triples)
        # print("\n\n")
        # print(right_triples)

    def assert_operator(self, data, add_data=None):
        raise Exception("This operator should not exists")

    def networkx(self):
        G = nx.MultiDiGraph()
        for s, t, r in self.edges:
            assert self.node2obj[s.id] == s
            assert self.node2obj[t.id] == t
            G.add_edge(s.id, t.id, rel_type=r)
        return G

    def feature(self, featurizer: FeaturizerBase):
        """_summary_
        Prerequisite for this function is the networkx call (this should happen during initializaition)
        Args:
            featurizer (Featurizer): _description_
        """
        self.node_features = {}
        for n_id in self.nodes:
            self.node2obj[n_id]
            self.node_features[n_id] = {
                "node_features": featurizer.featurize(self.node2obj[n_id])
            }

        nx.set_node_attributes(self.G, self.node_features)

    def to_dgl(self):
        """_summary_
        Prerequisite is a call to feature
        Returns:
            _type_: _description_
        """
        try:
            dgl_graph = dgl.from_networkx(
                self.G, edge_attrs=["rel_type"], node_attrs=["node_features"]
            )
        except Exception:
            dgl_graph = dgl.from_networkx(self.G, node_attrs=["node_features"])
            dgl_graph.edata["rel_type"] = th.tensor(np.array([]), dtype=th.int64)
            # print(self.data)
            # print(self.path)
            # exit()
        dgl_graph = dgl.add_self_loop(dgl_graph)
        return dgl_graph

    def get_nodes_in_edges(self):
        nodes = set()
        for node1, node2, _ in self.edges:
            nodes.add(node1.id)
            nodes.add(node2.id)
        return list(nodes)

    def add_sngl_trp_rel(self):
        nodes = self.get_nodes_in_edges()
        for t in self.triples:
            if not t.id in nodes:
                self.edges.append((t, t, 10))
