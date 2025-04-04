import json
import pandas as pd
import os
import json5
import pathlib

from graph_construction.qp.visitor.DefaultVisitor import DefaultVisitor
from graph_construction.qp.visitor.AbstractVisitor import dispatch

class DatExtractor(DefaultVisitor):
    def __init__(self):
        self.preds = {}
        self.ents = {}
        self.cur_qp_file = None  # used to keep track of wher e files are coming from.

    def dispatch(self, qp: dict, queryFile: str):
        """Custom dispatch for class

        Args:
            qp (dict): is a query plan and should be on that specific format.
        """
        self.cur_qp_file = queryFile
        dispatch(qp, self)

    def update(self, dct, key):
        try:
            dct[key].append(self.cur_qp_file)
        except KeyError:
            lst = [self.cur_qp_file]
            dct[key] = lst

    def visitTriple(self, input):
        if "http" in input["Predicate"]:
            self.update(self.preds, input["Predicate"])
        if "http" in input["Subject"]:
            self.update(self.ents, input["Subject"])
        if "http" in input["Object"]["value"]:
            self.update(self.ents, input["Object"]["value"])

    def visitTriplePath(self, input):
        if "Predicates" in input.keys() and "http" in input["Predicates"][0]:
            self.update(self.preds, input["Predicates"][0])
        if "http" in input["Subject"]:
            self.update(self.ents, input["Subject"])
        if "http" in input["Object"]["value"]:
            self.update(self.ents, input["Object"]["value"])
            

class UnseenQueryExtractor:
    """This class is responsible for extracting queries with unseen relations and entities
    """
    def __init__(self, path) -> None:
        self.path = path
        
        #set by set_train_pred_ents
        self.train_ents, self.train_preds = None, None
        
        #set by set_val_pred_ents
        self.val_ents, self.val_preds = None, None
        
        #set by set_train_pred_queryIDs
        self.train_pred_queryIDs = None
        
        #set by set_train_ent_queryIDs
        self.train_ent_queryIDs = None
        
        #set by set_val_pred_queryIDs
        self.val_pred_queryIDs = None
        
        #set by set_val_ent_queryIDs
        self.val_ent_queryIDs = None
        
        #set byset_test_pred_ents
        self.test_ents, self.test_preds = None, None
    
        #set by set_test_pred_queryIDs
        self.test_pred_queryIDs = None
        
        #set by set_test_ent_queryIDs
        self.test_ent_queryIDs = None
        
    def get_unseen_pred_queryIds(self):
        pred_queries = set()
        for x in self.test_preds.keys():
            if x in self.train_preds.keys():
                continue
            for q in self.test_preds[x]:
                pred_queries.add(q)
        return pred_queries
    
    def get_unseen_ent_queryIds(self):
        ent_queries = set()
        for x in self.test_ents.keys():
            if x in self.train_ents.keys():
                continue
            for q in self.test_ents[x]:
                ent_queries.add(q)
        return ent_queries
    
    def get_fraction_unseen(self):
        ent = (len([x for x in self.test_ents.keys() if x not in self.train_ents.keys()])/len(list(self.test_ents.keys()))) *100
        pred = (len([x for x in self.test_preds.keys() if x not in self.train_preds.keys()])/len(list(self.test_preds.keys()))) *100
        return ent, pred
    
    def set_train_pred_queryIDs(self):
        self.train_pred_queryIDs = self.get_pred_queryIDs(self.train_preds)
    
    def set_test_pred_queryIDs(self):
        self.test_pred_queryIDs = self.get_pred_queryIDs(self.test_preds)
        
    def set_val_pred_queryIDs(self):
        self.val_pred_queryIDs = self.get_pred_queryIDs(self.val_preds)
    
    
    def get_pred_queryIDs(self, preds):
        pred_queries = set()
        for x in preds.keys():
            for y in preds[x]:
                pred_queries.add( pathlib.Path(y).name)
        return pred_queries
    
    def set_train_ent_queryIDs(self):
        self.train_ent_queryIDs = self.get_ent_queryIDs(self.train_ents)
    
    def set_val_ent_queryIDs(self):
        self.val_ent_queryIDs = self.get_ent_queryIDs(self.val_ents)
    
    def set_test_ent_queryIDs(self):
        self.test_ent_queryIDs = self.get_ent_queryIDs(self.test_ents)
    
    def get_ent_queryIDs(self, ents):
        ent_queries = set()
        for x in ents.keys():
            for y in ents[x]:
                ent_queries.add( pathlib.Path(y).name)
        return ent_queries
    
    def get_train_pred_ents(self):
        """Extracts query plan files predicates and entities in training
        """
        trainExtractor = DatExtractor()
        files = UnseenQueryExtractor.get_files(self.path, train=True, val = False, test=False)[0]
        trainExtractor = UnseenQueryExtractor.iterate_files(files, trainExtractor)
        return trainExtractor.ents, trainExtractor.preds
        
    def get_val_pred_ents(self):
        """Extracts query plan files predicates and entities in training
        """
        trainExtractor = DatExtractor()
        files = UnseenQueryExtractor.get_files(self.path, train=False, val = True, test=False)[0]
        trainExtractor = UnseenQueryExtractor.iterate_files(files, trainExtractor)
        return trainExtractor.ents, trainExtractor.preds
    
    def set_train_pred_ents(self):
        """ Saves extracted query plan files predicates and entities in training in self.test_ents, and self.test_preds fields
        """
        self.train_ents, self.train_preds = self.get_train_pred_ents()
        
    def set_val_pred_ents(self):
        """ Saves extracted query plan files predicates and entities in training in self.test_ents, and self.test_preds fields
        """
        self.val_ents, self.val_preds = self.get_val_pred_ents()
        
    def get_test_pred_ents(self):
        """Extracts query plan files predicates and entities in training
        """
        testExtractor = DatExtractor()
        files = UnseenQueryExtractor.get_files(self.path, train=False, val = False, test=True)[0]
        testExtractor = UnseenQueryExtractor.iterate_files(files, testExtractor)
        return testExtractor.ents, testExtractor.preds
    
    def set_test_pred_ents(self):
        """ Saves extracted query plan files predicates and entities in test set in self.test_ents, and self.test_preds fields
        """
        self.test_ents, self.test_preds = self.get_test_pred_ents()
    
    def iterate_files(files, visitor:DatExtractor):
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
            visitor.dispatch(qp, f)
        return visitor
    
    def get_files(path, train=True, val=False, test = True):
        qp_path = f'{path}/queryplans'
        train_files, val_files, test_files = None, None, None
        if train:
            df = pd.read_csv(f'{path}/train_sampled.tsv', sep='\t')
            train_ids = list(df['id'].apply(lambda x : x[20:]))
            train_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if not '.' in x and x in train_ids]
        if val:
            df = pd.read_csv(f'{path}/val_sampled.tsv', sep='\t')
            val_ids = list(df['id'].apply(lambda x : x[20:]))
            val_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if not '.' in x and x in val_ids]
        if test:
            df = pd.read_csv(f'{path}/test_sampled.tsv', sep='\t')
            test_ids = list(df['id'].apply(lambda x : x[20:]))
            test_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if not '.' in x and x in test_ids]
        ret = []
        for i in [train_files, val_files, test_files]:
            if i is not None:
                ret.append(i)
        return tuple(ret)
        
class CompletelyUnseenQueryVisitor(DefaultVisitor):
    def __init__(self, train_preds, train_ents):
        self.preds = train_preds
        self.ents = train_ents
        self.unseen=True
        self.cur_qp_file = None  # used to keep track of wher e files are coming from.

    def dispatch(self, qp: dict, queryFile: str):
        """Custom dispatch for class

        Args:
            qp (dict): is a query plan and should be on that specific format.
        """
        self.cur_qp_file = queryFile
        self.unseen=True
        dispatch(qp, self)

    def visitTriple(self, input):
        if "http" in input["Predicate"]:
            if input["Predicate"] in self.preds:
                self.unseen=False
        if "http" in input["Subject"]:
            if input["Subject"] in self.ents:
                self.unseen=False
        if "http" in input["Object"]["value"]:
            if input["Object"]["value"] in self.ents:
                self.unseen=False

    def visitTriplePath(self, input):
        if "Predicates" in input.keys() and "http" in input["Predicates"][0]:
            if input["Predicates"][0] in self.preds:
                self.unseen=False
        if "http" in input["Subject"]:
            if input["Subject"] in self.ents:
                self.unseen=False
        if "http" in input["Object"]["value"]:
            if input["Object"]["value"] in self.ents:
                self.unseen=False


class CompletelyUnseenQueryExtractor:
    """This class is responsible for extracting queries with unseen relations and entities
    """
    def __init__(self, path) -> None:
        self.path = path
        
        #set by set_train_pred_ents
        self.train_ents, self.train_preds = None, None
        
        #set by set_train_pred_queryIDs
        self.train_pred_queryIDs = None
        
        #set by set_train_ent_queryIDs
        self.train_ent_queryIDs = None
        
        #set byset_test_pred_ents
        self.test_ents, self.test_preds = None, None
    
        #set by set_test_pred_queryIDs
        self.test_pred_queryIDs = None
        
        #set by set_test_ent_queryIDs
        self.test_ent_queryIDs = None
    
    def run(self):
        train_ents, train_preds = self.get_train_pred_ents()
        v = CompletelyUnseenQueryVisitor(train_preds, train_ents)
        files = CompletelyUnseenQueryExtractor.get_files(self.path, train=False, val=False, test=True)[0]
        q_files = CompletelyUnseenQueryExtractor.iterate_files(files,v)
        return q_files
        
    def get_unseen_pred_queryIds(self):
        pred_queries = set()
        for x in self.test_preds.keys():
            if x in self.train_preds.keys():
                continue
            for q in self.test_preds[x]:
                pred_queries.add(q)
        return pred_queries
    
    def get_unseen_ent_queryIds(self):
        ent_queries = set()
        for x in self.test_ents.keys():
            if x in self.train_ents.keys():
                continue
            for q in self.test_ents[x]:
                ent_queries.add(q)
        return ent_queries
    
    def set_train_pred_queryIDs(self):
        self.train_pred_queryIDs = self.get_pred_queryIDs(self.train_preds)
    
    def set_test_pred_queryIDs(self):
        self.test_pred_queryIDs = self.get_pred_queryIDs(self.test_preds)
    
    def get_pred_queryIDs(self, preds):
        pred_queries = set()
        for x in preds.keys():
            for y in preds[x]:
                pred_queries.add( pathlib.Path(y).name)
        return pred_queries
    
    def set_train_ent_queryIDs(self):
        self.train_ent_queryIDs = self.get_ent_queryIDs(self.train_ents)
    
    def set_test_ent_queryIDs(self):
        self.test_ent_queryIDs = self.get_ent_queryIDs(self.test_ents)
    
    def get_ent_queryIDs(self, ents):
        ent_queries = set()
        for x in ents.keys():
            for y in ents[x]:
                ent_queries.add( pathlib.Path(y).name)
        return ent_queries
    
    def get_train_pred_ents(self):
        """Extracts query plan files predicates and entities in training
        """
        trainExtractor = DatExtractor()
        files = UnseenQueryExtractor.get_files(self.path, train=True, val = False, test=False)[0]
        trainExtractor = UnseenQueryExtractor.iterate_files(files, trainExtractor)
        return trainExtractor.ents, trainExtractor.preds
    
    def set_train_pred_ents(self):
        """ Saves extracted query plan files predicates and entities in training in self.test_ents, and self.test_preds fields
        """
        self.train_ents, self.train_preds = self.get_train_pred_ents()
        
    def get_test_pred_ents(self):
        """Extracts query plan files predicates and entities in training
        """
        testExtractor = DatExtractor()
        files = UnseenQueryExtractor.get_files(self.path, train=False, val = False, test=True)[0]
        testExtractor = UnseenQueryExtractor.iterate_files(files, testExtractor)
        return testExtractor.ents, testExtractor.preds
    
    def set_test_pred_ents(self):
        """ Saves extracted query plan files predicates and entities in test set in self.test_ents, and self.test_preds fields
        """
        self.test_ents, self.test_preds = self.get_test_pred_ents()
    
    def iterate_files(files, visitor:CompletelyUnseenQueryVisitor):
        q_files = []
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
            visitor.dispatch(qp, f)
            if visitor.unseen:
                q_files.append(f)
        return q_files
    
    def get_files(path, train=True, val=False, test = True):
        qp_path = f'{path}/queryplans'
        train_files, val_files, test_files = None, None, None
        if train:
            df = pd.read_csv(f'{path}/train_sampled.tsv', sep='\t')
            train_ids = list(df['id'].apply(lambda x : x[20:]))
            train_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if not '.' in x and x in train_ids]
        if val:
            df = pd.read_csv(f'{path}/val_sampled.tsv', sep='\t')
            val_ids = list(df['id'].apply(lambda x : x[20:]))
            val_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if not '.' in x and x in val_ids]
        if test:
            df = pd.read_csv(f'{path}/test_sampled.tsv', sep='\t')
            test_ids = list(df['id'].apply(lambda x : x[20:]))
            test_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if not '.' in x and x in test_ids]
        ret = []
        for i in [train_files, val_files, test_files]:
            if i is not None:
                ret.append(i)
        return tuple(ret)

