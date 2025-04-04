import subprocess, os
import json
import sys

import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import write_dot
from matplotlib import pyplot as plt
import itertools as it

DISABLE_WARNING=True
try:
    PATH_JAR = os.environ["QG_JAR"] #'/qp/target/qp-1.0-SNAPSHOT.jar'
    PATH_JARQPP = os.environ["QPP_JAR"]
except KeyError as e:
    print(e)
    print('Please provide path to jar file for QP construction: "QG_JAR"')
    sys.exit(-1)

import jpype
import jpype.imports
from jpype.types import *
jpype.startJVM(classpath=[PATH_JAR,PATH_JARQPP])
from com.org import App

if DISABLE_WARNING:
    App.disableWarns()
def get_query_graph(query):
    return json.loads(str(App.getQueryGraph(query)))


def get_query_graph_nx(query:str, to_dot=None):
    qg = get_query_graph(query)
    G = nx.MultiDiGraph()
    node2label = {}
    for n in qg['nodes']:
        n_id = n['nodeId']
        n_type = n['nodeType']
        node2label[n_id] = f"{n_type}_{n_id}"

    for e in qg['edges']:
        G.add_edge(node2label[e[0]], node2label[e[1]], edge_type=e[2])
    if to_dot != None:
        write_dot(G,f'{to_dot}.dot')
        os.system(f"dot -Tsvg {to_dot}.dot > {to_dot}.svg")
    return G


def get_ent_rel(query):
    qg = get_query_graph(query)
    rels = set()
    ents = set()
    for n in qg['nodes']:
        if n['nodeType'] == 'PP' or n['nodeType'] == 'TP':
            if 'http' in n['subject']:
                ents.add(n['subject'])
            if 'http' in n['object']:
                ents.add(n['object'])
            if n['nodeType'] == 'PP':
                for p in n['predicateList']:
                    if 'http' in p:
                        if p.startswith('<'):
                            p = p[1:]
                        if p.endswith('>'):
                            p = p[:-1]
                        rels.add(p)
            else: #will always be TP
                p = n['predicate']
                if 'http' in p:
                    if p.startswith('<'):
                        p = p[1:]
                    if p.endswith('>'):
                        p = p[:-1]
                    rels.add(p)
    return ents, rels

def check_PP(query):
    qg = get_query_graph(query)
    is_PP = False
    for n in qg['nodes']:
        if n['nodeType'] == 'PP':
            is_PP = True
            break
    return is_PP

def check_filter(query):
    qg = get_query_graph(query)
    is_filter = False
    for n in qg['nodes']:
        if n['nodeType'] == 'FILTER':
            is_filter = True
            break
    return is_filter

def check_optional(query):
    qg = get_query_graph(query)
    is_Optional = False
    for e in qg['edges']:
        if e[2] == 'Optional' or e[2].lower() == "optional":
            is_Optional = True
            break
    return is_Optional

class BaselineFeatureExtract:
    def __init__(self):
        import jpype
        import jpype.imports
        from new_distance import GEDCalculator
        self.calc = GEDCalculator()
        from otheralgebra import AlgebraFeatureExtractor
        self.alg_extract = AlgebraFeatureExtractor()

        from semanticweb.sparql.preprocess import ReinforcementLearningExtractor
        self.extra = ReinforcementLearningExtractor()


    def distance_ged(self, query1:str, query2:str):
        try:
            return self.calc.calculateDistance(query1,query2)
        except:
            return np.inf
    def algebra_feat(self, query:str):
        return np.array(self.alg_extract.extractFeatures(query))