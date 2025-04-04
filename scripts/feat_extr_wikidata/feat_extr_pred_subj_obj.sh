url=http://172.21.233.14:8891/sparql/
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

echo "Script finnished after $SECONDS s"
