# Feature Extraction
Prerequisite: have the rdf store storing KG available.
## KG Stats
The KG stats are collected as:
```
URL=http://ENDPOINT_URL:8892/sparql
KGSTATFOLDER=/PlanRGCN/data/dbpedia2016 # where to store the extracted stat
KGSTATFOLDER=/data/planrgcn_feat/extracted_features_dbpedia2016 # where to store the extracted stat

mkdir -p "$KGSTATFOLDER"/predicate/batches
mkdir -p "$KGSTATFOLDER"/entity
#predicate features
python3 -m feature_extraction.predicates.pred_util extract-predicates -e $URL --output_dir $KGSTATFOLDER
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat -e $URL --input_dir $KGSTATFOLDER --output_dir "$KGSTATFOLDER" --batch_start 1 --time_log pred_freq_time_2.log --batch_end -1
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat-sub-obj -e $URL --input_dir $KGSTATFOLDER --output_dir "$KGSTATFOLDER" --batch_start 1 --time_log pred_stat_subj_obj_time_2.log --batch_end -1

#entity Features
#Missign one here
python3 -m feature_extraction.entity.entity_util extract-entity-stat -e $URL --input_dir "$KGSTATFOLDER"/entity --output_dir "$KGSTATFOLDER"/entity --ent_file "$KGSTATFOLDER"/entity/entities.json --batch_start 1 --time_log ent_stat_time.log --batch_end -1

#literal Features
python3 -m feature_extraction.literal_utils distinct-literals -e $URL --output_dir "$KGSTATFOLDER"/literals --lits_file "$KGSTATFOLDER"/literals/literals.json
python3 -m feature_extraction.literal_utils extract-lits-stat -e $URL --output_dir "$KGSTATFOLDER"/literals --lits_file "$KGSTATFOLDER"/literals/literals.json --time_log lit_stat_time_1.log --batch_start 1 --batch_end -1 --timeout 1200
python3 -m feature_extraction.literal_utils extract-lits-statv2 -e $URL --output_dir "$KGSTATFOLDER"/literals --lits_file "$KGSTATFOLDER"/literals/literals.json --time_log lit_stat_time_30.log --batch_start 1 --batch_end -1 --timeout 1200
```