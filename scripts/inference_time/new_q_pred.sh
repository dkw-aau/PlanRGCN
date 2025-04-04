
'
pip3 install ray[tune]
pip3 install jpype1
pip3 install -e /PlanRGCN/utils/
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
'

#Property paths
NEW_QUERY_FOLDER=/data/DBpedia_3_class_full/newPPs

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
prep_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/prepper.pcl"
model_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/best_model.pt"
config_path="/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/model_config.json"
output_path="${NEW_QUERY_FOLDER}/plan_inference.csv"
plan_inference


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