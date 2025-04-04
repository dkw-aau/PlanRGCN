
url=http://130.225.39.154:8890/sparql
url=http://172.21.233.14:8891/sparql
output_dir=/PlanRGCN/extracted_features_wd/entities
mkdir -p $output_dir
#python3 /PlanRGCN/feature_extraction/feature_extraction/entity/extract_entity.py $url $output_dir
#echo "Entities extracted after $SECONDS"
task=extract-entity-stat
output_dir=/PlanRGCN/extracted_features_wd/entities
pred_file=/PlanRGCN/extracted_features_wd/entities/entities.json
input_dir=/PlanRGCN/extracted_features_wd/entities
batch_start=1
batch_end=-1



python3 -m feature_extraction.entity.entity_util $task -e $url --output_dir $output_dir --ent_file $pred_file --batch_start $batch_start --batch_end $batch_end

url=http://172.21.233.23:8890/sparql/
