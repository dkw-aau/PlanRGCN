from analysis.util import plot_operator_presence
import pandas as pd

df = pd.read_csv('/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_statv2.tsv', sep='\t')
plot_operator_presence(df, showPlot=False, plotname='/qpp/DBpediea2016_operator_presence.png',x_label ='')