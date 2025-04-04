 #Unseen quereis
NEW_QUERY_FOLDER=/data/DBpedia_3_class_full/newUnseenQs3
NEW_QUERY_FOLDER=/data/DBpedia_3_class_full/newUnseenQs4

query_path="${NEW_QUERY_FOLDER}/queries.tsv"
plan_inference () {

python3 /PlanRGCN/inference.py \
  --prep_path "$prep_path" \
  --model_path "$model_path" \
  --config_path "$config_path" \
  --output_path "$output_path" \
  --query_path "$query_path"
}

# DBpedia Run
base="/data/DBpedia_3_class_full/planRGCNWOpredCo"
base="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co"
base="/data/DBpedia_3_class_full/plan16_10_2024_4096_2048"

prep_path="${base}/prepper.pcl"
model_path="${base}/best_model.pt"
#best f1 score model
#model_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/ray_save/train_function_2024-06-12_13-09-48/train_function_0bc66_00000_0_batch_size=256,dropout=0.0000,l1=4096,l2=1024,loss_type=cross-entropy,lr=0.0000,pred_com_path=pred2in_2024-06-12_13-09-48/checkpoint_000020/checkpoint.pt"
config_path="${base}/model_config.json"
output_path="${NEW_QUERY_FOLDER}/plan_inference.csv"
#plan_inference


#assimes that a queries.tsv is availble in new queries folder
bs_inference() {
  python3 -c """
import pandas as pd
train = pd.read_csv('$DATA_DIR/train_sampled.tsv',sep='\t')
val = pd.read_csv('$DATA_DIR/val_sampled.tsv',sep='\t')
newQs = pd.read_csv('$query_path',sep='\t')
df = pd.concat([train, val, newQs])
df.to_csv('$QUERY_LOG', sep='\t', index =False)
"""
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

K=25
QUERY_LOG="${NEW_QUERY_FOLDER}/all.tsv"
bs_inference