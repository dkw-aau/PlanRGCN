import json
from graph_construction.qp.visitor.AbstractVisitor import dispatch
from graph_construction.qp.visitor.DefaultVisitor import DefaultVisitor
import json5

class UnseenExtractorBase(DefaultVisitor):
    def __init__(self):
        self.preds = list()
        self.ents = list()
        
    def visitTriple(self,input):
        if 'http' in input['Predicate']:
            self.preds.append(input['Predicate'])
        if 'http' in input['Subject']:
            self.ents.append(input['Subject'])
        if 'http' in  input['Object']['value']:
            self.ents.append(input['Object']['value'])
        elif not input['Object']['value'].startswith('?'):
            self.lits.append(input['Object']['value'])
            
    def visitPath(self,input):
        if 'Predicates' in input.keys() and 'http' in input['Predicates'][0]:
            self.preds.append(input['Predicates'][0])
        if 'http' in  input['Subject']:
            self.ents.append(input['Subject'])
        if 'http' in  input['Object']['value']:
            self.ents.append(input['Object']['value'])
        elif not input['Object']['value'].startswith('?'):
            self.lits.append(input['Object']['value'])

def visitQueryPlans(files, visitor):
    for f in files:
        try:
            qp = json.load(open(f, "r"))
        except Exception:
            try:
                qp = json5.load(open(f, "r"))
            except Exception as e:
                print(f)
                print(e)
                break
        dispatch(qp,visitor)
    return visitor
class UnseenExtractorID(UnseenExtractorBase):
    """Extract the queries and file paths of queries that 

    Args:
        UnseenExtractorBase (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.ent_query_ids = list()
        self.pred_query_ids = list()
        self.lit_query_ids = list()
        
def retrieveQueriesUnseen(files, testVisitor, trainVisitor:UnseenExtractorBase):
    """_summary_

    Args:
        files (_type_): _description_
        testVisitor (_type_): _description_
        trainVisitor (_type_): _description_

    Returns:
        _type_: _description_
    """
    for f in files:
        try:
            qp = json.load(open(f, "r"))
        except Exception:
            try:
                qp = json5.load(open(f, "r"))
            except Exception as e:
                print(f)
                print(e)
                break
        dispatch(qp,visitor)
    return visitor