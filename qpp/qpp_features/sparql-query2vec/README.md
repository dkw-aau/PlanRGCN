# SPARQL Feature Representation
This is from https://github.com/dacasals/sparql-query2vec

With modification to make work with dataset.


## Approaches to vectorize Sparql queries and their metadata for Machine Learning tasks.
### Some baselines
- Rhassan code for algebra queries representation.
- Rhassan code for graph pattern queries representation based in kmedoids.
- **Queries representations using sets of features for DeepSet architecture**(working on..)

### Run
For run use that

System.out.println("Try with some of this parameters:");
```$bash
 java -jar file.jar kmedoids /path/to/input.csv /path/to/output.csv /path/to/ids_time.csv #-of-centroids
```
#### For generate edit distances vectors
```$bash
java -jar file.jar edit-distance /path/to/input.csv /path/to/output.csv /path/to/prefixes #-of-cores
```
Last number(4) is the number of cores or proccess to run in paralell.

#### For generate vectors for deepset architecture:
```$bash
java -jar file.jar deepset-features /path/to/input.csv /path/to/output.csv tables,joins,predicates /path/to/prefixes [--cores=numOfCores] [--length=numOfTuples] [--output-delimiter=symbolToDelimitColumns]
```

            
### Compilation
Execute to compile:
```$bash
mvn clean package
```
In the generated target you will find a graph-edit-distance-1.0-SNAPSHOT.jar

## Code for generating WDBench features:
```
mvn clean package -f pom.xml
jarpath=/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar
java -jar $jarpath algebra-features /SPARQLBench/bgp_opts.tsv /SPARQLBench/wdbench/bgp_opts_w_alg_stat.tsv --isWDBench=true
```
