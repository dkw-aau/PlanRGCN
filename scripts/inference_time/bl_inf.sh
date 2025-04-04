
#We first calculate the base inference times such that we can compare the computation time per baseline


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
process_inference() {
python3 -c """
import pandas as pd
q_df = pd.read_csv('$NEW_QUERY_FOLDER/queries.tsv', sep='\t')
f = open('$NEW_QUERY_FOLDER/alg_inf.txt','r')
txt = f.read()
alg_total_time = float(txt.split(':')[1])
f.close()
f = open('$NEW_QUERY_FOLDER/extra_inference.txt','r')
extra_total_time = float(txt.split(':')[1])
f.close()
ged = pd.read_csv('$NEW_QUERY_FOLDER/ged_dur.csv')

#Baseline Features
mean_ged_time = ged['time'].mean()
mean_alg_feat = alg_total_time/(len(q_df))

#NN inference time
nn = pd.read_csv('$NEW_QUERY_FOLDER/nn_prediction.csv')
mean_inf_nn = nn['inference'].mean()
#SVM inference time
svm = pd.read_csv('$NEW_QUERY_FOLDER/svm_pred.csv')
mean_inf_svm = svm['inference'].mean()

info = f'''Inference Statistics:
  Features:
    mean_ged_time: {mean_ged_time}
    mean_alg_time: {mean_alg_feat}
  Inference:
    mean SVM: {mean_inf_svm}
    mean NN: {mean_inf_nn}
  Total:
    NN: {mean_ged_time+mean_alg_feat+mean_inf_nn}
    SVM: {mean_ged_time+mean_alg_feat+mean_inf_svm}
'''
print(info)
with open('baseline_inference_summary.txt', 'w') as w:
  w.write(info)
"""
}

# We start with DBpedia.
DATA_DIR=/data/DBpedia_3_class_full
SVMTRAINER=/data/DBpedia_3_class_full/svm/svmtrainer.pickle
NNTRAINER=/data/DBpedia_3_class_full/nn/k25/nntrainer.pickle

BL_JAR=/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar
GED_query_file=/data/DBpedia_3_class_full/baseline/knn25/ged_queries.txt
QUERY_LOG=/data/DBpedia_3_class_full/all.tsv
K=25
NEW_QUERY_FOLDER=/data/DBpedia_3_class_full/test_inf

#We first setup the folder structure for this with the test queries
inference_time
process_inference

# We now proceed with Wikidata
DATA_DIR=/data/wikidata_3_class_full
SVMTRAINER=/data/wikidata_3_class_full/svm/svmtrainer.pickle
NNTRAINER=/data/wikidata_3_class_full/nn/k25/nntrainer.pickle

BL_JAR=/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar
GED_query_file=/data/wikidata_3_class_full/baseline/knn25/ged_queries.txt
QUERY_LOG=/data/wikidata_3_class_full/all.tsv
K=25
NEW_QUERY_FOLDER=/data/wikidata_3_class_full/test_inf

#We first setup the folder structure for this with the test queries
inference_time
process_inference