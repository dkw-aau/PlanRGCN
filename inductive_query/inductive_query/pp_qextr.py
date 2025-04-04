import json
import pandas as pd
import os
import json5
import pathlib

from graph_construction.qp.visitor.DefaultVisitor import DefaultVisitor
from graph_construction.qp.visitor.AbstractVisitor import dispatch

class PPExtractor(DefaultVisitor):
    def __init__(self):
        self.cur_qp_file = None  # used to keep track of wher e files are coming from.
        self.pp_files = []

    def dispatch(self, qp: dict, queryFile: str):
        """Custom dispatch for class

        Args:
            qp (dict): is a query plan and should be on that specific format.
        """
        self.pp = False
        self.cur_qp_file = queryFile
        dispatch(qp, self)
        if self.pp == True:
            self.pp_files.append(queryFile)

    def update(self, dct, key):
        try:
            dct[key].append(self.cur_qp_file)
        except KeyError:
            lst = [self.cur_qp_file]
            dct[key] = lst

    def visitTriple(self, input):
        pass

    def visitTriplePath(self, input):
        if "Predicates" in input.keys() and "http" in input["Predicates"][0]:
            self.pp = True
            

class PPQueryExtractor:
    """This class is responsible for extracting queries with unseen relations and entities
    """
    def __init__(self, path) -> None:
        self.path = path
        
        #set by set_test_pp
        self.test_pp = []
    
    def set_test_pp(self):
        self.test_pp = self.get_test_PP_files()
    
    def get_test_PP_files(self):
        trainExtractor = PPExtractor()
        files = PPQueryExtractor.get_files(self.path, train=False, val = False, test=True)[0]
        trainExtractor = PPQueryExtractor.iterate_files(files, trainExtractor)
        return trainExtractor.pp_files
    
    def get_train_PP_files(self):
        trainExtractor = PPExtractor()
        files = PPQueryExtractor.get_files(self.path, train=True, val = False, test=False)[0]
        trainExtractor = PPQueryExtractor.iterate_files(files, trainExtractor)
        return trainExtractor.pp_files
    
    def get_val_PP_files(self):
        trainExtractor = PPExtractor()
        files = PPQueryExtractor.get_files(self.path, train=False, val = True, test=False)[0]
        trainExtractor = PPQueryExtractor.iterate_files(files, trainExtractor)
        return trainExtractor.pp_files
    
    def iterate_files(files, visitor:PPExtractor):
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
            test_files = [f"{qp_path}/{x}" for x in os.listdir(qp_path) if x in test_ids]
        ret = []
        for i in [train_files, val_files, test_files]:
            if i is not None:
                ret.append(i)
        return tuple(ret)