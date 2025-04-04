import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class QueryLogImporterBase:
    def __init__(self):
        self.tsv_file = None
        self.query_df = None

    def get_query_log(self):
        raise NotImplementedError
class TSVFileQLImporter(QueryLogImporterBase):
    def __init__(self, tsv_file:str):
        self.tsv_file = tsv_file
        self.query_df = pd.read_csv(tsv_file, sep='\t') if tsv_file != None else None
    def get_query_log(self):
        return self.query_df