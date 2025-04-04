import os, json
import networkx as nx
import pandas as pd



def retrieveBGPs(query):
    s = 'java -jar '+path+' triples '+'\"' +query +'\"'
    bgps = os.popen(s).read()
    bgps = bgps.split('§§')
    bgps = [retrieveTriples(x) for x in bgps if x != '\n']
    return bgps

#requires lsq_extractor jar
def retrieveTriples(tripleString):
    triples = tripleString.split('\n')
    if len(triples) == 0:
        return []
    triples = [json.loads(x) for x in triples if x != '']
    return triples

def createTriplePatternGraph(BGPs):
    graph = nx.DiGraph()
    variables = set()
    for triples in BGPs:
        for trp in triples:
            if trp['subject'].startswith('?'):
                variables.add(trp['subject'])
            if trp['object'].startswith('?'):
                variables.add(trp['object'])
            if trp['predicate'].startswith('?'):
                variables.add(trp['predicate'])
            graph.add_node(trp['subject'])
            graph.add_node(trp['predicate'])
            graph.add_node(trp['object'])
            graph.add_edge(trp['subject'],trp['predicate'])
            graph.add_edge(trp['subject'],trp['object'])
    return graph, variables

def preprocess_query(query_string):
    query_string=query_string.replace('""','"').replace('"','§')
    return query_string

def pattern_types(G, variables):
    no_stars, no_paths, hybrid, sink = 0,0,0,0
    for var in variables:
        no_in_edges = len(G.in_edges(var))
        no_out_edges = len(G.out_edges(var))
        if no_in_edges == 0 and no_out_edges > 1:
            no_stars +=1
        if no_in_edges == 1 and no_out_edges == 1:
            no_paths+= 1
        if no_in_edges >=1 and no_out_edges >= 1 and (no_in_edges+no_out_edges) >= 3:
            hybrid+=1
        if no_out_edges == 0 and no_in_edges >= 2:
            sink+=1
    return no_stars,no_paths,hybrid,sink

def create_pattern_types(query):
    bgps = retrieveBGPs(query)
    G, variables = createTriplePatternGraph(bgps)
    no_stars,no_paths,hybrid,sink = pattern_types(G, variables)
    return no_stars,no_paths,hybrid,sink

def modify_df(df):
    no_stars = []
    no_paths = []
    no_hybrids = []
    no_sink = []
    for i in range(0,len(df)):
        query_string = df.iloc[i]['queryString']
        #print("At: {}, query_string: {}".format(i,query_string))
        s,p,h,si=create_pattern_types(query_string)
        no_stars.append(s)
        no_paths.append(p)
        no_hybrids.append(h)
        no_sink.append(si)
    df['stars'] = no_stars
    df['paths'] = no_paths
    df['hybrids'] = no_hybrids
    df['sinks'] = no_sink
    return df
def modify_df_multiproce():
    from multiprocessing import Pool
def open_csv(path):
    df = pd.read_csv(path, sep='§', engine='python')
    df.set_index('queryID')
    df['queryString']=df['queryString'].apply(lambda x: preprocess_query(x))
    return df


query = """SELECT  *
  WHERE
     { ?s  <foo:test>   ?p ;
           <foo:test2>  ?z
       OPTIONAL
         { ?x  <foo:test>   ?var1 ;
               <foo:test3>  ?var2
         }
     }
"""
if __name__ == "__main__":
    path = os.environ['lsq_extractor']
    bgps = retrieveBGPs(query)
    G,variables =createTriplePatternGraph(bgps)
    no_stars,no_paths,hybrid,sink = pattern_types(G,variables)


