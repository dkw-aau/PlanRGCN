
def dispatch(input, visitor):
    match input['opName']:
        case "project":
            for op in input['subOp']:
                dispatch(op,visitor)
        case "BGP":
            visitor.visitBGP(input)
        case "Triple":
            visitor.visitTriple(input)
        case "path":
            visitor.visitTriplePath(input)
        case "sequence":
            for op in input['subOp']:
                dispatch(op,visitor)
        case "filter":
            visitor.visitFilter(input)
        case "conditional":
            visitor.visitOptional(input)
        case "leftjoin":
            visitor.visitOptional(input)
        case "extend":
            for op in input['subOp']:
                dispatch(op,visitor)
        case "group":
            for op in input['subOp']:
                dispatch(op,visitor)
        case "distinct":
            for op in input['subOp']:
                dispatch(op,visitor)
        case "table":
            for op in input['subOp']:
                dispatch(op,visitor)
        case "assign":
            visitor.visitDefault(input)
        case "disjunction":
            visitor.visitDisjunction(input)
        case "join":
            visitor.visitJoin(input)
        case "reduced":
            visitor.visitDefault(input)
        case "service":
            visitor.visitDefault(input)
        case "graph":
            visitor.visitDefault(input)
        case _:
            raise Exception(f"unsupported {input['opName']}\n{input}")

class AbstractVisitor:
    def visitBGP(self, input):
        pass
    def visitTriple(self, input):
        pass
    def visitFilter(self, input):
        pass
    def visitOptional(self, input):
        pass
    def visitDefault(self, input):
        for op in input['subOp']:
            dispatch(op,self)
    def visitDisjunction(self, input):
        pass
    def visitJoin(self, input):
        pass

class PrintVisitor(AbstractVisitor):
    def visitBGP(self, input):
        print("starting ", input['opName'])
        for e in input['subOp']:
            dispatch(e, self)
        print("ending ", input['opName'])
        
    def visitTriple(self,input):
        print("starting ", input['opName'])
        print(input)
        print("ending ", input['opName'])
        
    def visitFilter(self, input):
        print("starting ", input['opName'])
        print("expression ", input['expr'])
        for o in input['subOp']:
            dispatch(o, self)
        print("ending ", input['opName'])
        
    def visitOptional(self, input):
        print("starting ", input['opName'])
        for o in input['subOp']:
            dispatch(o, self)
        print("ending ", input['opName'])
    def visitDisjunction(self, input):
        print("starting ", input['opName'])
        for o in input['subOp']:
            dispatch(o, self)
        print("ending ", input['opName'])
    
    def visitJoin(self, input):
        print("starting ", input['opName'])
        for o in input['subOp']:
            dispatch(o, self)
        print("ending ", input['opName'])
