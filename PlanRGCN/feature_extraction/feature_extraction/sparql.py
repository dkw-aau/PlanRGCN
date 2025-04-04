

from SPARQLWrapper import SPARQLWrapper, JSON, POST
import time

from feature_extraction.res_proc import SelProcUtil


class Endpoint:
    def __init__(self, endpoint_url):
        self.sparql = SPARQLWrapper(endpoint_url, defaultGraph='http://localhost:8890/dataspace')
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
    def run_query(self, query_str: str):
        if not hasattr(self,'sparql'):
            print('SPARQL Endpoint has not been initialised!!!')
            exit()
        try:
            self.sparql.setQuery(query_str)
        except Exception:
            print("Query could not be executed!")
            return query_str
        results = self.sparql.query().convert()
        return results
    def time_and_run_query(self, query_str: str):
        if not hasattr(self,'sparql'):
            print('SPARQL Endpoint has not been initialised!!!')
            exit()
        start = time.time()
        try:
            self.sparql.setQuery(query_str)
        except Exception as e:
            print("Query could not be executed!")
            return e, None
        results = self.sparql.query().convert()
        duration = time.time() - start
        return results, duration

    def run_query_and_results(self, query_str):
        res = self.run_query(query_str)
        return SelProcUtil.get_bindings(res)
