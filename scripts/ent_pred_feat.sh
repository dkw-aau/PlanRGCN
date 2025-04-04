#/PlanRGCN/qpe/pom.xml mvn package -f /PlanRGCN/qpe/pom.xml
: 'jar_path=/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar
wikidata_log=/SPARQLBench/wikidata_lsq/ordered_queries.tsv
wdbench_log=/SPARQLBench/wdbench/bgp_opts.tsv
dbpedia2016_log=/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_stat.tsv

java -jar $jar_path extract-predicates-query-log $dbpedia2016_log /PlanRGCN/extracted_features_dbpedia2016/predicate/predicate_in_dbpedia2016.json
java -jar $jar_path extract-predicates-query-log $wdbench_log /PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log_1.json
java -Xmx1g -jar $jar_path extract-predicates-query-log $wikidata_log /PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log_2.json
python3 -c """
import json
data1 = json.load(open('/PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log_1.json'))
data2 = json.load(open('/PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log_2.json'))
data1.extend(data2)
data1 = list(set(data1))
json.dump(data1,open('/PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log.json','w'))
"""
rm /PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log_1.json
rm /PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log_2.json
'
wikidata_preds=/PlanRGCN/extracted_features_wd/predicate/predicate_in_wikidata_log.json
dbpedia2016_preds=/PlanRGCN/extracted_features_dbpedia2016/predicate/predicate_in_dbpedia2016.json
: 'python3 -c """
import json
def count_preds(p):
 preds = json.load(open(p,'r'))
 print(len(preds))
count_preds('$wikidata_preds')
count_preds('$dbpedia2016_preds')
"""'
wikidata_output=/PlanRGCN/extracted_features_wd
dbpedia2016_output=/PlanRGCN/extracted_features_dbpedia2016
quantile=0.9
wikidata_url=http://130.225.39.154:8891/sparql
dbpedia_url=http://130.225.39.154:8890/sparql
#Wikidata terminated at 742: due to endpoint closed
python3 /PlanRGCN/feature_extraction/feature_extraction/pred_ent/pred_ent_features.py $wikidata_url $wikidata_output $wikidata_preds $quantile
#python3 /PlanRGCN/feature_extraction/feature_extraction/pred_ent/pred_ent_features.py $dbpedia_url $dbpedia2016_output $dbpedia2016_preds $quantile
