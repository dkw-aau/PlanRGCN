#!/bin/bash

# $1 = http://172.21.233.23:84/lsq/sparql
# $2 = 8000                                 -maximum amount of queires
# $3 = /qpp/queries2015_2016.tsv 

(cd ../qpp_features/sparql-query2vec/ && mvn package && mvn install:install-file -Dfile=target/sparql-query2vec-0.0.1.jar -DpomFile=pom.xml )
(cd ../lsq_extract/ && mvn package)
if [-n $4]
then
    (cd ../lsq_extract/ && java -jar target/lsq_extract-1.0-SNAPSHOT.jar final-interval $1 $2 $3)
fi
(cd ../lsq_extract/workload_gen/ && python3 duplicate_removal.py $3 $3_clean)
(iconv -f utf-8 -t ascii -c $3_clean > $3_clean2)
mv $3_clean2 $3_clean
(java -jar ../lsq_extract/target/lsq_extract-1.0-SNAPSHOT.jar workload-stat $3_clean $3_clean_w_stats)
(cd ../lsq_extract/workload_gen/ && python3 duplicate_removal.py $3_clean_w_stats $3_clean_w_stats)
(python3 ../lsq_extract/workload_gen/bin_queries.py)