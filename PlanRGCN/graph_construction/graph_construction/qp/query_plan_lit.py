from graph_construction.qp.query_plan import *
from graph_construction.node import Node, FilterNode, TriplePattern2

class QueryPlanLit(QueryPlan):
    def __init__(self, data):
        super().__init__(data)

    def add_triple(self, data, add_data=None):
        t = TriplePattern2(data)
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

