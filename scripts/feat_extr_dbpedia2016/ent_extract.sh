

url=http://130.225.39.154:8890/sparql
output_dir=/PlanRGCN/extracted_features_dbpedia2016/entities
mkdir -p $output_dir
#python3 /PlanRGCN/feature_extraction/feature_extraction/entity/extract_entity.py $url $output_dir

pred_file="$output_dir"/entities.json
input_dir=/PlanRGCN/extracted_features_dbpedia2016/entities
task=extract-entity-stat
batch_start=1
batch_end=-1
python3 -m feature_extraction.entity.entity_util $task -e $url --output_dir $output_dir --ent_file $pred_file --batch_start $batch_start --batch_end $batch_end

echo "Script finnished after $SECONDS s"