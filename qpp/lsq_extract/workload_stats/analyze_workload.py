import subprocess, pandas as pd
import ast

PATH_TO_JAR = "jars/sparql-query2vec-0.0.1.jar"

query = "PREFIX  dbpo: <http://dbpedia.org/ontology/> PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX  xsd:  <http://www.w3.org/2001/XMLSchema#> PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX  foaf: <http://xmlns.com/foaf/0.1/> PREFIX  dbprop: <http://dbpedia.org/property/>  SELECT  * WHERE   { ?city     rdf:type    dbpo:Place ;               rdfs:label  §西冷§@en .     ?airport  rdf:type    dbpo:Airport       { ?airport  dbpo:city  ?city }     UNION       { ?airport  dbpo:location  ?city }     UNION       { ?airport  dbprop:cityServed  ?city }     UNION       { ?airport  dbpo:city  ?city }       { ?airport  dbprop:iata  ?iata }     UNION       { ?airport  dbpo:iataLocationIdentifier  ?iata }     OPTIONAL       { ?airport  foaf:homepage  ?airport_home }     OPTIONAL       { ?airport  rdfs:label  ?name }     OPTIONAL       { ?airport  dbprop:nativename  ?airport_name }     FILTER ( ( ! bound(?name) ) || langMatches(lang(?name), §zh§) )   } "


def extract_alg_op_counts(query):
    query = "\""+query.replace("§","\\\"") + "\""
    #subprocess.call([f"java -jar {PATH_TO_JAR}", "algebra-feature-query", f"--query={query}"])
    t= ast.literal_eval( subprocess.check_output(f"java -jar {PATH_TO_JAR} algebra-feature-query --query={query}", shell=True, universal_newlines=True))
    return t

#good for manual evaluation
def query_stat_gen(path):
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        yield extract_alg_op_counts(line)

#remebmer to change no_queries
def query_stat_lst(path, no_queries= 50000):
    lst = []
    with open(path, "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        temp = extract_alg_op_counts(line)
        print(f"At {idx} of {len(lines)}")
        if type(temp) is dict:
            lst.append( extract_alg_op_counts(line))
        if idx == no_queries:
            return lst
    return lst

def alg_stat_to_csv(path, output_path):
    lst = query_stat_lst(path)
    df = pd.DataFrame(lst)
    df.to_csv(output_path)
    return df

if __name__ == "__main__":
    df = alg_stat_to_csv("query_workload.txt", "alg_stats.csv")
