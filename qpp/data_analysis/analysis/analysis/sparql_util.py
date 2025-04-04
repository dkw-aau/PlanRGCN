from SPARQLWrapper import SPARQLWrapper, JSON

def create_sparql(url='http://172.21.233.14:8891/sparql/'):
    sparql = SPARQLWrapper(url)
    sparql.setReturnFormat(JSON)
    return sparql

def query(query, sparql):
    sparql.setQuery(query)
    try:
        ret = sparql.queryAndConvert()
        return len(ret['results']['bindings'])
    except Exception as e:
        print(e)