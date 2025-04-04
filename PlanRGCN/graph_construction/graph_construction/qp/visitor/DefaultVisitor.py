from graph_construction.qp.visitor.AbstractVisitor import AbstractVisitor
from graph_construction.qp.visitor.AbstractVisitor import dispatch

class DefaultVisitor(AbstractVisitor):
    def visitBGP(self, input):
        for e in input['subOp']:
            dispatch(e, self)
        
    def visitTriple(self,input):
        pass
    
    def visitTriplePath(self,input):
        pass

    def visitExpr(self, input):
        pass
        
    def visitFilter(self, input):
        self.visitExpr(input['expr'])
        for o in input['subOp']:
            dispatch(o, self)
        
    def visitOptional(self, input):
        for o in input['subOp']:
            dispatch(o, self)
            
    def visitDisjunction(self, input):
        for o in input['subOp']:
            dispatch(o, self)
    
    def visitJoin(self, input):
        for o in input['subOp']:
            dispatch(o, self)