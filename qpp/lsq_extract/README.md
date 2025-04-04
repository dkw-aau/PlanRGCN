# LSQ Data Extractor

The purpose of this part of the code is to extract the logs and other relevant information for benchmark creation (SPARQL QPP).
## Dependencies
Requires Java with Maven support & Python to run.

## Dataset
The used queries are LSQ Query logs available at 
The used files are the following:
- [bench-dbpedia-20151124-lsq2.nt.bz2](https://hobbitdata.informatik.uni-leipzig.de/lsqv2/dumps/dbpedia/bench-dbpedia-20151124-lsq2.nt.bz2)
- [bench-dbpedia.3.5.1](https://hobbitdata.informatik.uni-leipzig.de/lsqv2/dumps/dbpedia/bench-dbpedia.3.5.1.log-lsq2.nt.bz2)

### Environment Variables
```
export lsq_extractor=$(pwd)/target/lsq_extract-1.0-SNAPSHOT-jar-with-dependencies.jar
```
## Extract LSQ data from endpoint
```
java -jar lsq_extract-1.0-SNAPSHOT-jar-with-dependencies.jar extract $(file_path)
```

## Filter Valid LSQ queries.
```
java -jar lsq_extract-1.0-SNAPSHOT-jar-with-dependencies.jar legal-query-check /Users/abirammohanaraj/Documents/GitHub/lsq_extract/lsq_data.csv /Users/abirammohanaraj/Documents/GitHub/lsq_extract/legal_lsq_data.csv /Users/abirammohanaraj/Documents/GitHub/lsq_extract/illegal_lsq_data.csv
```
## cleaning of data
/qpp/lsq_extract/workload_gen/utils.py
clean_query_strings()

## Torch in java
NOTE: this should be moved away from this repository.
```
export PYTORCH_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/lib
export PYTORCH_VERSION=1.13.1
export PYTORCH_FLAVOR=cpu
```

## Maven commands
```
mvn install:install-file -Dfile=clust4j-1.2.4-SNAPSHOT.jar -DgroupId=com.clust4j -DartifactId=clust4j -Dversion=1.2.4 -Dpackaging=jar
mvn install:install-file -Dfile=GMT.jar -DgroupId=com.example -DartifactId=gmt -Dversion=1.0 -Dpackaging=jar
mvn install:install-file -Dfile=target/sparql-query2vec-0.0.1.jar -DpomFile=pom.xml 

```
## To extract data
```
java -jar target/lsq_extract-1.0-SNAPSHOT.jar final-interval http://172.21.233.23:84/lsq/sparql 8000 /qpp/queries2015_2016.tsv
duplicate removal.py after
```