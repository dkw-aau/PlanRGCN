# Feature Extraction

## Prerequisite:
A SPARQL endpoint with the KG needs to be accessible.

Input Field:
```
WIKIDATA=/PlanRGCN/data/wikidata
URL=http://172.21.233.23:8891/sparql
```
## Predicate Feature Extraction
### Predicates
```
WIKIDATA=/PlanRGCN/data/wikidata
python3 -m feature_extraction.predicates.pred_util extract-predicates -e $URL --output_dir $WIKIDATA
```
### Predicate Co-Occurence Features
```
mkdir "$WIKIDATA"/predicate/batches
python3 -m feature_extraction.predicates.pred_util extract-co-predicates \
-e $URL \
--input_dir $WIKIDATA \
--output_dir "$WIKIDATA"/predicate \
--batch_start 1 --batch_end -1
```

### Predicate Statistics
Freq, ent, lits:
```
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat -e http://172.21.233.23:8891/sparql --input_dir $WIKIDATA --output_dir "$WIKIDATA" --batch_start 1 --time_log pred_freq_time_1.log --batch_end 86
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat -e http://172.21.233.23:8891/sparql --input_dir $WIKIDATA --output_dir "$WIKIDATA" --batch_start 86 --time_log pred_stat_time_2.log --batch_end -1
```
Subj, obj
```
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat-sub-obj -e http://172.21.233.23:8891/sparql --input_dir $WIKIDATA --output_dir "$WIKIDATA" --batch_start 1 --time_log pred_stat_subj_obj_time_1.log --batch_end 86
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat-sub-obj -e http://172.21.233.23:8891/sparql --input_dir $WIKIDATA --output_dir "$WIKIDATA" --batch_start 86 --time_log pred_stat_subj_obj_time_2.log --batch_end -1
```

### Entities
```
WIKIDATA=/PlanRGCN/data/wikidata
python3 -m feature_extraction.entity.extract_entity http://172.21.233.23:8891/sparql "$WIKIDATA"/entity
```

### Entity Statitistics
```
WIKIDATA=/PlanRGCN/data/wikidata
URL=http://172.21.233.23:8891/sparql
python3 -m feature_extraction.entity.entity_util extract-entity-stat \
-e $URL \
--input_dir "$WIKIDATA"/entity \
--output_dir "$WIKIDATA"/entity \
--ent_file /PlanRGCN/data/wikidata/entity/entities.json \
--batch_start 1 \
--time_log ent_stat_time.log \
--batch_end 140

python3 -m feature_extraction.entity.entity_util extract-entity-stat \
-e $URL \
--input_dir "$WIKIDATA"/entity \
--output_dir "$WIKIDATA"/entity \
--ent_file /PlanRGCN/data/wikidata/entity/entities.json \
--batch_start 140 \
--time_log ent_stat_time.log \
--batch_end -1
```

### Literals

```
WIKIDATA=/PlanRGCN/data/wikidata
URL=http://172.21.233.23:8891/sparql
python3 -m feature_extraction.literal_utils distinct-literals \
-e $URL \
--output_dir "$WIKIDATA"/literals \
--lits_file "$WIKIDATA"/literals/literals.json
```

```
python3 -m feature_extraction.literal_utils extract-lits-stat \
-e $URL \
--output_dir "$WIKIDATA"/literals \
--lits_file "$WIKIDATA"/literals/literals.json \
--time_log lit_stat_time_1.log \
--batch_start 1 \
--batch_end 1049

python3 -m feature_extraction.literal_utils extract-lits-stat \
-e $URL \
--output_dir "$WIKIDATA"/literals \
--lits_file "$WIKIDATA"/literals/literals.json \
--time_log lit_stat_time_2.log \
--batch_start 1049 \
--batch_end -1
```
