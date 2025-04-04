import ast

class TimeLoader:
    def load_lit_stat(file, dct):
        s_type = 'lit_stat'
        with open(file,'r') as f:
            for line in f:
                s = line.split(',')
                match s[1]:
                    case " freq":
                        dct[s_type]['freq'].append(ast.literal_eval(s[2]))
        return dct
        
    def load_lit_extract(file, dct):
        with open(file,'r') as f:
            for line in f:
                if ':' in line:
                    dct['lit_extract'] = ast.literal_eval(line.split(':')[1])
        return dct
        
    def load_ent_stat(file, dct):
        s_type = 'ent_stat'
        with open(file,'r') as f:
            for line in f:
                s = line.split(',')
                match s[1]:
                    case " obj":
                        dct[s_type]['obj'].append(ast.literal_eval(s[2]))
                    case " subj":
                        dct[s_type]['subj'].append(ast.literal_eval(s[2]))
                    case " freq":
                        dct[s_type]['freq'].append(ast.literal_eval(s[2]))
        return dct
        
    def load_ent_extract(file, dct):
        with open(file,'r') as f:
            for line in f:
                if ':' in line:
                    dct['ent_extract'] = ast.literal_eval(line.split(':')[1])
        return dct
        
    def load_pred_stat(file, dct):
        with open(file,'r') as f:
            for line in f:
                s = line.split(',')
                match s[1]:
                    case " lits":
                        dct['pred_stat']['lits'].append(ast.literal_eval(s[2]))
                    case " ents":
                        dct['pred_stat']['ents'].append(ast.literal_eval(s[2]))
                    case " freq":
                        dct['pred_stat']['freq'].append(ast.literal_eval(s[2]))
                    case " subj":
                        dct['pred_stat']['subj'].append(ast.literal_eval(s[2]))
                    case " obj":
                        dct['pred_stat']['obj'].append(ast.literal_eval(s[2]))
        return dct
    def load_pred_extract(file, dct):
        with open(file,'r') as f:
            for line in f:
                if ':' in line:
                    dct['pred_extract'] = ast.literal_eval(line.split(':')[1])
        return dct
        
    def load_pred_co_extract(file, dct):
        with open(file,'r') as f:
            for line in f:
                if ':' in line:
                    dct['pred_stat']['pred_co_extract'].append( ast.literal_eval(line.split(':')[1]))
        return dct
        
    def load_pred_louvain(file, dct):
        with open(file,'r') as f:
            for line in f:
                s = line.split(',')
                dct[s[0]]= ast.literal_eval(s[1])
        return dct