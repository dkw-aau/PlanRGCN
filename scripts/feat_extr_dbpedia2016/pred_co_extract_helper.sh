url=http://172.21.233.23:8890/sparql
pred_file=predicates.json
task=extract-co-predicates
input_dir=/PlanRGCN/extracted_features
output_dir=/PlanRGCN/extracted_features/predicate
batch_start=1679
batch_end=3356
python3 -m feature_extraction.predicates.pred_util $task -e $url \
        --input_dir $input_dir \
        --output_dir $output_dir \
        --pred_file $pred_file \
        --batch_start $batch_start \
        --batch_end $batch_end
