url=http://130.225.39.154:8890/sparql
task=extract-predicates
output_dir=/PlanRGCN/extracted_features
pred_file=predicates.json

#python3 -m feature_extraction.predicates.pred_util $task -e $url --output_dir $output_dir --pred_file $pred_file
task=extract-co-predicates
input_dir=/PlanRGCN/extracted_features
output_dir=/PlanRGCN/extracted_features/predicate
batch_start=1
batch_end=1678
: 'python3 -m feature_extraction.predicates.pred_util $task -e $url \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --pred_file $pred_file \
        --batch_start $batch_start \
        --batch_end $batch_end
'
url=http://172.21.233.23:8890/sparql/
pred_file=predicate_in_dbpedia2016.json
task=extract-predicates-stat
input_dir=/PlanRGCN/extracted_features/predicate
output_dir=/PlanRGCN/extracted_features/predicate
batch_start=1
batch_end=13
python3 -m feature_extraction.predicates.pred_stat_feat $task -e $url \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --pred_file $pred_file \
        --batch_start $batch_start \
        --batch_end $batch_end

