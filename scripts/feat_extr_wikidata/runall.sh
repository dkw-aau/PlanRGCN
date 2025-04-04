url=http://172.21.233.23:8891/sparql
pred_file=predicates.json
task=extract-predicates-stat-sub-obj
input_dir=/PlanRGCN/extracted_features_wd
output_dir=/PlanRGCN/extracted_features_wd
batch_start=1
batch_end=-1
python3 -m feature_extraction.predicates.pred_stat_feat $task -e $url \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --pred_file $pred_file \
        --batch_start $batch_start \
        --batch_end $batch_end



exit
#original script
url=http://172.21.233.23:8891/sparql
task=extract-predicates
output_dir=/PlanRGCN/extracted_features_wd
pred_file=predicates.json
# mkdir -p /PlanRGCN/extracted_features_wd/predicate/batches

python3 -m feature_extraction.predicates.pred_util $task \
        -e $url \
        --output_dir $output_dir \
        --pred_file $pred_file


task=extract-co-predicates
input_dir=/PlanRGCN/extracted_features_wd
output_dir=/PlanRGCN/extracted_features_wd/predicate
batch_start=1
batch_end=10000000
python3 -m feature_extraction.predicates.pred_util $task -e $url \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --pred_file $pred_file \
        --batch_start $batch_start \
        --batch_end $batch_end
mkdir -p /PlanRGCN/extracted_features_wd/predicate/pred_co
mv /PlanRGCN/extracted_features_wd/predicate/batch_response/* /PlanRGCN/extracted_features_wd/predicate/pred_co

task=extract-predicates-stat
input_dir=/PlanRGCN/extracted_features_wd/predicate
output_dir=/PlanRGCN/extracted_features_wd/predicate
batch_start=1
batch_end=10000000
python3 -m feature_extraction.predicates.pred_stat_feat $task -e $url \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --pred_file $pred_file \
        --batch_start $batch_start \
        --batch_end $batch_end



task=extract-entity-query-log
queries=/SPARQLBench/wdbench/bgp_opts.tsv
outputPath=/PlanRGCN/extracted_features_wd/entities/entities_in_wikidata.json
mkdir -p /PlanRGCN/extracted_features_wd/entities/
mvn package -f /PlanRGCN/qpe/pom.xml
java -jar /PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar $task $queries $outputPath


task=extract-entity-stat
output_dir=/PlanRGCN/extracted_features_wd/entities
pred_file=/PlanRGCN/extracted_features_wd/entities/entities_in_wikidata.json
input_dir=/PlanRGCN/extracted_features_wd/entities
batch_start=1
batch_end=10000000


python3 -m feature_extraction.entity.entity_util $task -e $url --output_dir $output_dir --ent_file $pred_file --batch_start $batch_start --batch_end $batch_end

