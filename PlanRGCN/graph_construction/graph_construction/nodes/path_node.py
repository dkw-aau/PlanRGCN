from graph_construction.node import TriplePattern, is_variable_check,TriplePattern2
from graph_construction.node import Node
from graph_construction.qp.qp_utils import pathOpTypes


class PathNode(TriplePattern2):
    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class
        # Not implemented for now
        # self.path_predicates = list()
        # for p in data["Predicates"]:
        #    self.path_predicates.append(node_class(p))

        self.subject = node_class(data["Subject"])
        if 'Predicate Path' in data.keys():
            predicate = data['Predicate Path'].split('<')[1].split('>')[0]
        else:
            predicate = data["Predicates"][0]
        if isinstance(predicate, str):
            self.predicate = node_class(predicate)
            self.p_mod_max = 0
            self.p_mod_min = 0
        else:
            self.predicate = node_class(predicate["Predicate"])
            self.p_mod_max = predicate["min"]
            self.p_mod_min = predicate["max"]
        self.path_complexity: list[pathOpTypes] = list()
        if not "pathComplexity" in data.keys():
            self.path_complexity.append(pathOpTypes.get_path_op(data['pathType']))
        else:
            for comp in data["pathComplexity"]:
               self.path_complexity.append(pathOpTypes.get_path_op(comp))

        self.object = node_class(data["Object"]["value"])
        try:
            self.object.datatype = data["Object"]["datatype"]
        except KeyError:
            pass
        try:
            self.object.langtag = data["Object"]["langTag"]
        except KeyError:
            pass

        # with good results of 80% f1 score - old encoding - Not tested with path though
        self.subject.nodetype = 0
        self.predicate.nodetype = 1
        self.object.nodetype = 2

        # New variable encoding
        self.subject.nodetype = 0 if is_variable_check(self.subject.node_label) else 1
        self.predicate.nodetype = (
            0 if is_variable_check(self.predicate.node_label) else 1
        )
        self.object.nodetype = 0 if is_variable_check(self.object.node_label) else 1
        if "level" in data.keys():
            self.level = data["level"]
            self.depthLevel = self.level

    def __str__(self):
        return f"PATH ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"

    def __repr__(self):
        return f"PATH ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"



class PathNode2(PathNode):
    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class
        # Not implemented for now
        # self.path_predicates = list()
        # for p in data["Predicates"]:
        #    self.path_predicates.append(node_class(p))

        self.subject = node_class(data["subject"])

        self.predicate = node_class(data["predicateList"][0].split('<')[1].split('>')[0])
        self.p_mod_max = data['pred_max'] if data['pred_max'] != -1 else 0 #None
        self.p_mod_min = data['pred_min'] if data['pred_min'] != -1 else 0 #None
        self.path_complexity: list[pathOpTypes] = list()
        for comp in data["pathComplexity"]:
           self.path_complexity.append(pathOpTypes.get_path_op(comp))

        self.object = node_class(data["object"])
        if data['isLiteral']:
            self.object.datatype = data['objectDatatype']
            try:
                if data['objectLang'] != "":
                    self.object.langtag = data["objectLang"]
            except Exception:
                pass

        # with good results of 80% f1 score - old encoding - Not tested with path though
        self.subject.nodetype = 0
        self.predicate.nodetype = 1
        self.object.nodetype = 2

        # New variable encoding
        self.subject.nodetype = 0 if is_variable_check(self.subject.node_label) else 1
        self.predicate.nodetype = (
            0 if is_variable_check(self.predicate.node_label) else 1
        )
        self.object.nodetype = 0 if is_variable_check(self.object.node_label) else 1
        if "level" in data.keys():
            self.level = data["level"]
            self.depthLevel = self.level