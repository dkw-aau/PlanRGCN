from nltk import ngrams
import re
from analyzer import open_csv
import numpy as np
import kmedoids
import pandas as pd
import pickle as pcl
from multiprocessing import Pool, Process
import time,os

def query_to_ngram(query, n):
    query=  query.split(" WHERE ",1)[-1]
    query = replace_variables(query)
    query = re.sub(r'[^\w\s]', '', query)
    
    n_grams = ngrams(query.split(),n)
    return list(n_grams)

#replace variables with var (maybe this should be distinguished)
def replace_variables(query):
    split = query.split()
    reconstructed = ""
    for i in split:
        if i.startswith('?'):
            v = 'var'
        else:
            v = i
        reconstructed += (v+' ')
    return reconstructed[:-1]

def jaccard_old(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = len(list(set(list1).union(list2)))
    return float(intersection) / float(union)
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = len(set(list1)) +len(set(list2)) - intersection
    return float(intersection) / float(union)

def select_queries(path, path2):
    df = open_csv(path)
    df2 = open_csv(path2)
    df = pd.concat([df,df2], axis=0,ignore_index=True)
    describe_ids = []
    without_where = []
    for i in range(len(df)):
        if 'DESCRIBE' in df.iloc[i]['queryString']:
            describe_ids.append( df.iloc[i]['queryID'])
        elif len(df.iloc[i]['queryString'].split(' WHERE ')) <2:
            without_where.append( df.iloc[i]['queryID'])
    filters = describe_ids
    filters.extend(without_where)
    df = df[~df['queryID'].isin(filters)]
    df.to_csv('/data/workload_gen/processed_queries.csv', sep='§')
    return df

def dist_matrix(df, n):
    dists = {}
    for i in range(len(df)):
        dists[i,i] = 1
        i_ngram = query_to_ngram(df.iloc[i]['queryString'],n)
        for j in range(i+1, len(df)):
            j_ngram = query_to_ngram(df.iloc[j]['queryString'],n)
            sim = 1 - jaccard(i_ngram,j_ngram)
            dists[i,j] = sim
            dists[j,i] = sim
    return dists

def dist_sub_computation_write(data_arg):
  df = data_arg[0]
  n=data_arg[1]
  el_range=data_arg[2]
  distfolder=data_arg[3]
  query_grams = data_arg[4]
  #data = {'queryID':[], 'dists':[]}
  f = open("{}/distances_{}.txt".format(distfolder,str(el_range).replace(",","_")), "w")
  print("range"+str(el_range))
  for i in range(el_range[0],el_range[1]+1):
    #i_ngram = query_to_ngram(df.iloc[i]['queryString'],n)
    i_ngram=query_grams[i]
    dists =[]
    for j in range(len(df)):
      #j_ngram = query_to_ngram(df.iloc[j]['queryString'],n)
      j_ngram = query_grams[j]
      dist = 1 - jaccard(i_ngram,j_ngram)
      dists.append(dist)
    f.write(df.iloc[i]['queryID'])
    f.write(",")
    f.write(str(dists)+"\n")
    f.flush()
    #data['queryID'].append(df.iloc[i]['queryID'])
    #data['dists'].append(dists)
  
  f.close()

  pass
def precompute_ngrams(df, n):
  query_ngrams = []
  for i in range(len(df)):
    query_ngrams.append(query_to_ngram(df.iloc[i]['queryString'],n))
  return query_ngrams

def read_parralel_dists(path_dir):
  files = os.listdir(path_dir)


def dist_matrix_parralel(df,n, cores, distfolder):
  #processes = []
  if cores > len(df):
    print("Amount of cores compared to DataFrame is wrong. Please check the data again")
    exit()
  
  element_count = round(len(df)/cores)
  el_ranges = []
  prev = 0
  for i in range(cores):
    if prev+element_count > len(df):
      el_ranges.append((prev, len(df)-1))
    else:
      el_ranges.append((prev,prev+element_count))
      prev += element_count
      prev += 1
  start = time.time()
  query_grams = precompute_ngrams(df, 3)
  print("TIme to create {}-grams for queries: {}".format(n,time.time()-start))
  data = [(df, n,x, distfolder, query_grams) for x in el_ranges]
  p = Pool(cores)
  p.map(dist_sub_computation_write,data)
  """for i in range(cores):
    p = Process(target=dist_sub_computation_write,args=(df,(),distfolder))
    processes.append(p)
    p.start()
  
  #for p in processes:
  #  p.start()
  for p in processes:
    p.join()"""


def dist_matrix_write(df, n):
    dists = {}
    for i in range(len(df)):
        dists[i,i] = 1
        i_ngram = query_to_ngram(df.iloc[i]['queryString'],n)
        for j in range(i+1, len(df)):
            j_ngram = query_to_ngram(df.iloc[j]['queryString'],n)
            sim = 1 - jaccard(i_ngram,j_ngram)
            dists[i,j] = sim
            dists[j,i] = sim
    return dists
def dist_dict_to_np(dict,n):
    result = np.zeros((n,n))
    for key, val in dict.items():
        result[key[0]][key[1]] = val
    return result
def pickle_object(obj,path):
    pickler = open(path,"wb")
    pcl.dump(obj,pickler)
    pickler.close()

class NgramProcessor:
  
  def __init__(self, n):
    self.vocab = {}
    self.inverse_vocab = {}
    self.n = n
    self.vocabCounter = 0
  
  def add_to_vocab(self, query:str):
    grams = query_to_ngram(query, self.n)
    for gram in grams:
      if self.vocab.get(gram) == None:
        self.vocab[gram] = self.vocabCounter
        self.inverse_vocab[self.vocabCounter] = gram
        self.vocabCounter += 1

  def save(self, path):
    with open(path,'wb') as f:
      pcl.dump(self,f)
  
  

  def initialize_df(self,df, log=True):
    start = time.time()
    for i in range(len(df)):
      self.add_to_vocab(df.iloc[i]['queryString'])
    if log:
      print(f"DataFrame initialized in {time.time()-start}")

def load_ngram_processor(path):
    with open(path, 'rb') as f:
      proc = pcl.load(f)
    return proc

if __name__ == "__main__":
    query ="""?city     rdf:type    dbpo:Place ;
                rdfs:label  "Champagne"@nn .
     ?airport  rdf:type    dbpo:Airport
        { ?airport  dbpo:city  ?city }
      UNION
        { ?airport  dbpo:location  ?city }
      UNION
        { ?airport  dbprop:cityServed  ?city }
      UNION
        { ?airport  dbpo:city  ?city }
        { ?airport  dbprop:iata  ?iata }
      UNION
        { ?airport  dbpo:iataLocationIdentifier  ?iata }
      OPTIONAL
        { ?airport  foaf:homepage  ?airport_home }
      OPTIONAL
        { ?airport  rdfs:label  ?name }
      OPTIONAL
        { ?airport  dbprop:nativename  ?airport_name }
      FILTER ( ( ! bound(?name) ) || langMatches(lang(?name), "nn") )
    """
    #the following should be outcommented
    """start = time.time()
    df = select_queries('/data/workload_gen/lsq_queries/dbpedia2010_valid.csv', '/data/workload_gen/lsq_queries/legal_lsq_data.csv')
    print("TIme for loading data: {}".format(time.time()-start))
    start = time.time()
    dist_matrix_parralel(df,3,25,"/data/workload_gen/dists")
    print("TIme for creating distance files: {}".format(time.time()-start))"""
    
    #d = dist_matrix(df,3)
    #d = dist_dict_to_np(d,len(df))
    #c = kmedoids.fasterpam(d,10)
    #print(c)
    #print(c.loss)
    #print(c.medoids)
    #print(c.labels)
    #pickle_object(c,"/data/workload_gen/kmediod.pickle")
    
    #n_gram = query_to_ngram(query,3)
    #print(jaccard( n_gram,n_gram))
    #print(jaccard(n_gram,n_gram))
    """start = time.time()
    df = select_queries('/data/workload_gen/lsq_queries/dbpedia2010_valid.csv', '/data/workload_gen/lsq_queries/legal_lsq_data.csv')
    print(f"TIme for loading data: {time.time()-start}")
    """
    #processor = NgramProcessor(4)
    #processor.initialize_df(df)
    #processor.save("/data/workload_gen/ngramProc.pickle")
    processor = load_ngram_processor("/data/workload_gen/ngramProc.pickle")
    print(f"N Gram processor: {processor.vocab}")
    