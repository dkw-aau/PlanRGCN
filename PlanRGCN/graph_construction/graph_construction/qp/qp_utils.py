from graph_construction.node import FilterNode
from graph_construction.nodes.PathComplexException import PathComplexException
from graph_construction.stack import Stack
from enum import Enum


class pathOpTypes(Enum):
    ZERO_OR_MORE = 0
    ZERO_OR_ONE = 1
    ONE_OR_MORE = 2
    INVERSE = 3
    MOD = 4
    SEQ = 5

    # Not considererd for now
    ALTERNATIVE = 7
    NEGATIVE_PROP_SET = 7
    FIXED_LENGTH = 7
    MULTI = 7
    REVERSE = 7

    def get_max_operations():
        return 6

    def get_path_op(path_op: str):
        match path_op:
            case "P_ZeroOrOne":
                return pathOpTypes.ZERO_OR_ONE
            case "zeroOrOne":
                return pathOpTypes.ZERO_OR_ONE
            case "P_ZeroOrMore1":
                return pathOpTypes.ZERO_OR_MORE
            case "P_ZeroOrMoreN":
                return pathOpTypes.ZERO_OR_MORE
            case 'ZeroOrMore':
                return pathOpTypes.ZERO_OR_MORE
            case "P_OneOrMore1":
                return pathOpTypes.ONE_OR_MORE
            case "P_OneOrMoreN":
                return pathOpTypes.ONE_OR_MORE
            case "OneOrMore":
                return pathOpTypes.ONE_OR_MORE
            case "P_ReverseLink":
                return pathOpTypes.INVERSE
            case "P_Inverse":
                return pathOpTypes.INVERSE
            case "P_Mod":
                return pathOpTypes.MOD
            case "Mod":
                return pathOpTypes.MOD
            case "P_Seq":
                return pathOpTypes.SEQ
            case _:
                raise PathComplexException(
                    f"Property path operation {path_op} has not been considered!"
                )


def get_relation_types(trp1, trp2, common_variable):
    """return the relation type index for most the cases.

    Args:
        trp1 (TriplePattern): _description_
        trp2 (TriplePattern): _description_
        common_variable (str): variable common for trp1 and trp2

    Returns:
        int: relation type
    """
    # filter nodes
    if isinstance(trp2, FilterNode):
        return 9

    # s-s
    if trp1.subject == common_variable and trp2.subject == common_variable:
        return 0
    # s-p
    if trp1.subject == common_variable and trp2.predicate == common_variable:
        return 1
    # s-o
    if trp1.subject == common_variable and trp2.object == common_variable:
        return 2

    # p-s
    if trp1.predicate == common_variable and trp2.subject == common_variable:
        return 3
    # p-p
    if trp1.predicate == common_variable and trp2.predicate == common_variable:
        return 4
    # p-o
    if trp1.predicate == common_variable and trp2.object == common_variable:
        return 5
    # o-s
    if trp1.object == common_variable and trp2.subject == common_variable:
        return 6
    # o-p
    if trp1.object == common_variable and trp2.predicate == common_variable:
        return 7
    # o-s
    if trp1.object == common_variable and trp2.object == common_variable:
        return 8


class QueryPlanUtils:
    "filter rel definied in getjointype method"

    def get_relations(op):
        match op:
            case "conditional":
                return 10
            case "leftjoin":
                return 10
            case "join":
                return 11
        """match op:
            case "conditional":
                return 11
            case "leftjoin":
                return 12
            case "join":
                return 13
            case "union":
                return 14
            case "minus":
                return 15"""
        raise Exception("Operation undefind " + op)

    def extract_triples(data: dict):
        triple_data = []
        stack = Stack()
        stack.push(data)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current.keys():
                for node in reversed(current["subOp"]):
                    stack.push(node)
            if current["opName"] == "Triple":
                triple_data.append(current)
        return triple_data

    def extract_triples_filter(data: dict):
        triple_data = []
        stack = Stack()
        stack.push(data)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current.keys():
                for node in reversed(current["subOp"]):
                    stack.push(node)
            if current["opName"] == "Triple":
                triple_data.append(current)
        return triple_data

    def map_extracted_triples(triple_dct: list[dict], trpl_list: list):
        res_t = list()
        for t in trpl_list:
            if t in triple_dct:
                res_t.append(t)
        return res_t
