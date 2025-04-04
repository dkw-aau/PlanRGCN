import networkx as nx
from graph_construction.node import FilterNode, TriplePattern, TriplePattern2, TriplePattern3, FilterNode02
from graph_construction.nodes.PathComplexException import PathComplexException
from graph_construction.nodes.path_node import PathNode, PathNode2
from graph_construction.qp.query_plan import QueryPlan
from graph_construction.stack import Stack
from functools import partialmethod


class QueryPlanPath(QueryPlan):
    def __init__(self, data):
        QueryPlan.max_relations = 13
        QueryPlanPath.set_max_rels()
        super().__init__(data)

    def set_max_rels(number=13):
        QueryPlan.max_relations = number

    def process(self, data):
        self.level = 0
        self.data = data
        self.triples: list[TriplePattern2 | PathNode] = list()
        self.filters: list[FilterNode] = list()
        self.edges = list()

        self.join_vars = {}
        self.filter_dct = {}
        self.op_lvl = {}

        self.iterate(self.add_tripleOrPath)

        # self.filters_process()
        self.iterate_ops(self.add_filters, "filter")

        self.assign_trpl_ids()
        self.assign_filt_ids()

        # adds more complex operations
        # self.add_join()
        self.iterate_ops(self.add_binaryOP, "leftjoin")
        self.iterate_ops(self.add_binaryOP, "conditional")
        self.iterate_ops(self.add_binaryOP, "join")
        # self.add_leftjoin()
        # self.add_conditional()
        self.add_sngl_trp_rel()

        # print(self.edges)
        self.nodes = [x.id for x in self.triples]
        self.nodes.extend([x.id for x in self.filters])

        self.node2obj = {}
        self.initialize_node2_obj()
        self.G = self.networkx()
        self.G.add_nodes_from(self.nodes)

    def iterate(self, func):
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
                    self.iterate_bgp(current, func, None, filter=None, new=True)
                else:
                    self.level += 1
                    for node in reversed(current["subOp"]):
                        node["level"] = self.level
                        stack.push(node)
            func(current)

    def add_tripleOrPath(self, data, add_data=None):
        if data["opName"] == "path":
            try:
                t = PathNode(data)
            except KeyError:
                raise PathComplexException(f"Did not work for {data}")
        elif data["opName"] == "Triple":
            t = TriplePattern2(data)
        else:
            return
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

    def iterate_bgp_new(self, data, func, filter=None):
        self.level += 1
        for triple in data["subOp"]:
            triple["level"] = self.level
            func(triple, add_data=filter)

    def iterate_bgp(self, data, func, node_type, filter=None, new=False):
        if new:
            self.iterate_bgp_new(data, func, filter=filter)
            return

        self.level += 1
        for triple in data["subOp"]:
            triple["level"] = self.level
            if triple["opName"] == node_type:
                func(triple, add_data=filter)

    def iterate_ops(self, func, node_type: str):
        return super().iterate_ops(func, node_type)

    def add_binaryOP(self, data, add_data=None):
        return super().add_binaryOP(data, add_data)

    def add_filters(self, data, add_data=None):
        return super().add_filters(data, add_data)

    # add_join = partialmethod(iterate_ops, add_binaryOP, "join")
    add_leftjoin = partialmethod(iterate_ops, add_binaryOP, "leftjoin")
    add_conditional = partialmethod(iterate_ops, add_binaryOP, "conditional")
    filters_process = partialmethod(iterate_ops, add_filters, "filter")


def rel_lookup(rel):
    match rel:
        case "S_S":
            return 0
        case "S_P":
            return 1
        case "S_O":
            return 2
        case "P_S":
            return 3
        case "P_P":
            return 4
        case "P_O":
            return 5
        case "O_S":
            return 6
        case "O_P":
            return 7
        case "O_O":
            return 8
        case "filter":
            return 9
        case "OPTIONAL":
            return 10
        case "Optional":
            return 10
        case "SingleTripleOrCatesian":
            return 11
        case _:
            raise Exception("undefined "+ rel)


class QueryGraph(QueryPlanPath):
    def __init__(self, query_graph):
        super().__init__(query_graph)
        QueryPlan.max_relations = 13
        QueryPlanPath.set_max_rels()

    def set_max_rels(number=13):
        QueryPlan.max_relations = number

    def process(self, data):
        self.data = data
        self.triples: list[TriplePattern2 | PathNode] = list()
        self.filters: list[FilterNode] = list()

        self.nodes = list()
        self.edges = list()
        self.node2obj = {}
        for node in data['nodes']:
            match node['nodeType']:
                case "TP":
                    n = TriplePattern3(node)
                case "PP":
                    n = PathNode2(node)
                case "FILTER":
                    n = FilterNode02(node)
                case _:
                    raise Exception('Unknwn node' + node['nodeType'])
            nodeid = node['nodeId']
            self.nodes.append(nodeid)
            self.node2obj[nodeid] = n

        self.G = self.networkx()

    def networkx(self):
        G = nx.MultiDiGraph()
        for e in self.data['edges']:
            G.add_edge(e[0], e[1], rel_type=rel_lookup(e[2]))
        G.add_nodes_from(self.nodes)
        return G
