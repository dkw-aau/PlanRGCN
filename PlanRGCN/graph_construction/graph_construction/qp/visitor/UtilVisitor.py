from graph_construction.qp.visitor.DefaultVisitor import DefaultVisitor
from graph_construction.qp.visitor.AbstractVisitor import dispatch
import numpy as np

class JoinChecker(DefaultVisitor):
    def visitJoin(self, input):
        print(len(input['subOp']))
        print(input['subOp'][0])
        print('\n\n\n')
        print(input['subOp'][1])
        #for o in input['subOp']:
        #    dispatch(o, self)
        raise Exception()
        
class LiteralChecker(DefaultVisitor):
    def __init__(self):
        self.laTags = list()
        self.datatypes = list()
        
    def visitTriple(self,input):
        if 'datatype' in input['Object'].keys():
            self.datatypes.append(input['Object']['datatype'])
        if 'langTag' in input['Object'].keys():
            self.laTags.append(input['Object']['langTag'])


class LiteralsFeaturizer:
    las = ['', 'cr', 'am', 'fo', 'fr', 'ast', 'mg', 'cs', 'th', 'yo', 'cbk-zam', 'eng', 'jv', 'et', 'vi', 'sq', 'cdo', 'ru', 'no', 'ka', 'arc', 'bi', 'nl', 'cy', 'ckb', 'en-US', 'ja', 'bg', 'az', 'lb', 'ur', 'pt', 'wa', 'pl', 'hy', 'kk', 'ab', 'so', 'ko', 'an', 'br', 'ace', 'ro', 'ay', 'as', 'ga', 'gsw', 'ar', 'ce', 'it', 'ca', 'es', 'fa', 'frr', 'bpy', 'war', 'en-ca', 'be-tarask', 'mk', 'en', 'arz', 'udm', 'sk', 'sv', 'oc', 'de', 'da', 'bar', 'li', 'bs', 'km', 'rmy', 'zh', 'ceb', 'bo', 'gu', 'uk', 'nb']
    da_types = ['http://www.w3.org/2001/XMLSchema#boolean', 'http://example.org/datatype#specialDatatype', 'http://www.w3.org/2001/XMLSchema#string', 'http://www.w3.org/2001/XMLSchema#integer', 'http://www.w3.org/2001/XMLSchema#decimal', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString']
    
    def feat_size():
        return len(LiteralsFeaturizer.las) + len(LiteralsFeaturizer.da_types)
    
    def lang_feat(la):
        idx = -1
        for en, i in enumerate(LiteralsFeaturizer.las):
            if i == la:
                idx = en
        vec = np.zeros(len(LiteralsFeaturizer.las))
        if idx != -1:
            vec[idx] = 1
        return vec
        
    def dataype_feat(data_type):
        idx = -1
        for en, i in enumerate(LiteralsFeaturizer.da_types):
            if i == data_type:
                idx = en
        vec = np.zeros(len(LiteralsFeaturizer.da_types))
        if idx != -1:
            vec[idx] = 1
        return vec
    
    def feat(obj):
        try:
            lang_vec = LiteralsFeaturizer.lang_feat(obj['langTag'])
        except Exception:
            lang_vec = LiteralsFeaturizer.lang_feat(None)
        try:
            type_vec = LiteralsFeaturizer.dataype_feat(obj['datatype'])
        except Exception:
            type_vec = LiteralsFeaturizer.dataype_feat(None)
        return np.concatenate([lang_vec, type_vec], axis=0)
    
#feats = LiteralsFeaturizer()
if __name__ == "__main__":
    obj = {'value': '101860', 'datatype': 'http://www.w3.org/2001/XMLSchema#string',}
    print(LiteralsFeaturizer.feat(obj).shape)
        