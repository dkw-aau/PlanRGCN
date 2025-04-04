
from graph_construction.node import TriplePattern
from graph_construction.qp.qp_utils import QueryPlanUtils

from graph_construction.qp.query_plan import QueryPlan

class QueryPlanCommonBi(QueryPlan):
    """This query plan class only adds relations between binary BGP operations with the triple patterns share the same variables

    Args:
        QueryPlan (_type_): _description_
    """

    max_relations = 16

    def __init__(self, data) -> None:
        super().__init__(data)

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
            for l in left_triples:
                # consider adding the other way for union as a special case
                if self.is_triple_common(l, r):
                    self.edges.append(
                        (r, l, QueryPlanUtils.get_relations(data["opName"]))
                    )

    def is_triple_common(self, l: TriplePattern, r: TriplePattern):
        """checks if the triple patterns are joinable

        Args:
            l (TriplePattern): left triple pattern
            r (TriplePattern): right triple pattern
        """
        for lvar in l.get_joins():
            for rvar in r.get_joins():
                if lvar == rvar:
                    return True
        return False

    def add_self_loop_triples(self):
        for t in self.triples:
            self.edges.append((t, t, 15))