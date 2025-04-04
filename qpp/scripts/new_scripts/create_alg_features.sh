# $1 = /qpp/dataset/
mvn -f '/PlanRGCN/qpp/qpp_features/sparql-query2vec/pom.xml' package
(cd /qpp/qpp_features/sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar algebra-features "$1/train_sampled.tsv" "$1/train_alg.tsv")
(cd /qpp/qpp_features/sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar algebra-features "$1/val_sampled.tsv" "$1/val_alg.tsv")
(cd /qpp/qpp_features/sparql-query2vec/target && java -jar sparql-query2vec-0.0.1.jar algebra-features "$1/test_sampled.tsv" "$1/test_alg.tsv")