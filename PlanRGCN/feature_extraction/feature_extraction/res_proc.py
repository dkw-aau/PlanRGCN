"""
Utilities for processing results of SPARQLWrapper calls
"""

class SelProcUtil:

    @staticmethod
    def get_bindings(res:dict):
        return res['results']['bindings']
