inference_time() {
  mkdir -p $NEW_QUERY_FOLDER
  cp $DATA_DIR/test_sampled.tsv $NEW_QUERY_FOLDER
  mv $NEW_QUERY_FOLDER/test_sampled.tsv $NEW_QUERY_FOLDER/queries.tsv
  python3 /PlanRGCN/qpp/qpp_new/qpp_new/pred_n_qs.py \
      --svm_trainer_path $SVMTRAINER \
      --data_dir $DATA_DIR \
      --new_query_folder $NEW_QUERY_FOLDER \
      --jarfile $BL_JAR \
      --GED_query_file $GED_query_file \
      --query_log $QUERY_LOG \
      --K $K \
      --nn_trainer_path $NNTRAINER

  rm $NEW_QUERY_FOLDER/extra.*
}

# We start with DBpedia.
DATA_DIR=/data/DBpedia_3_class_full
SVMTRAINER=/data/DBpedia_3_class_full/svm/svmtrainer.pickle
NNTRAINER=/data/DBpedia_3_class_full/nn/k25/nntrainer.pickle

BL_JAR=/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar
GED_query_file=/data/DBpedia_3_class_full/baseline/knn25/ged_queries.txt
QUERY_LOG=/data/DBpedia_3_class_full/additionalPPtestQs/all.tsv
K=25
NEW_QUERY_FOLDER=/data/DBpedia_3_class_full/additionalPPtestQs

inference_time

old_inference () {

python3 /PlanRGCN/inference.py \
  --prep_path "$prep_path" \
  --model_path "$model_path" \
  --config_path "$config_path" \
  --output_path "$output_path" \
  --query_path "$query_path"
}

NEW_QUERY_FOLDER=/data/DBpedia_3_class_full/additionalPPtestQs
prep_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/prepper.pcl"
model_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/best_model.pt"
config_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/model_config.json"
output_path="/data/DBpedia_3_class_full/additionalPPtestQs/planrgcn_pred.csv"
query_path="$NEW_QUERY_FOLDER/queries.tsv"
old_inference