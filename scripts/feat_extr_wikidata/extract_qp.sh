queryplan=/PlanRGCN/extracted_features_wd/queryplans
querypath=/SPARQLBench/wdbench/bgp_opts.tsv

# mvn clean package -f /PlanRGCN/qpe/pom.xml

java -jar qpe/target/qpe-1.0-SNAPSHOT.jar extract-query-plans $querypath $queryplan
