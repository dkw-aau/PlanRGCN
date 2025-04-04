from SPARQLWrapper import SPARQLWrapper, JSON, POST


class Query:
    def __init__(self, endpoint_url=None):
        if not endpoint_url is None:
            self.sparql = SPARQLWrapper(endpoint_url)
            self.sparql.setReturnFormat(JSON)
            self.sparql.setMethod(POST)

    def run_query(self, query_str: str):
        if not hasattr(self, "sparql"):
            print("SPARQL Endpoint has not been initialised!!!")
            exit()
        self.query_str = query_str
        try:
            self.sparql.setQuery(query_str)
        except Exception:
            print("Query could not be executed!")
            exit()
        self.results = self.sparql.query().convert()
        return self.results
