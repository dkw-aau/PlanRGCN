tar -cvf wikidata.tar.gz /data/wikidata_3_class_full/test_sampled.tsv \
    /data/wikidata_3_class_full/train_sampled.tsv \
    /data/wikidata_3_class_full/val_sampled.tsv \
    /data/wikidataV2/test_sampled.tsv  /data/wikidataV2/train_sampled.tsv  /data/wikidataV2/val_sampled.tsv

tar -cvf dbpedia.tar.gz /data/DBpedia_3_class_full/test_sampled.tsv \
    /data/DBpedia_3_class_full/train_sampled.tsv \
    /data/DBpedia_3_class_full/val_sampled.tsv \
    /data/DBpediaV2/test_sampled.tsv  /data/DBpediaV2/train_sampled.tsv  /data/DBpediaV2/val_sampled.tsv

---
#Models

tar -cvf dbpedia_models.tar.gz \
    /data/DBpedia_3_class_full/svm \
    /data/DBpedia_3_class_full/nn/k25 \
    /data/DBpedia_3_class_full/baseline/knn25/ged_queries.txt \
    /data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co \


tar -cvf wikidata_models.tar.gz \
    /data/wikidata_3_class_full/svm \
    /data/wikidata_3_class_full/nn/k25 \
    /data/wikidata_3_class_full/baseline/knn25/ged_queries.txt \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/best_model.pt \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/model_config.json \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/objective.py \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/prepper.pcl \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/test_pred.csv \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/train_pred.csv \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/val_pred.csv \
    /data/wikidata_3_class_full/planRGCN_no_pred_co/train_time.log