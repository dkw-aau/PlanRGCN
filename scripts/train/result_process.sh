
# New result processing code

MyVar="${DEPLOY_ENV:-default_value}"



exit
#python3 /PlanRGCN/scripts/post_predict.py -s /data/DBpedia_3_class_full -t 5 -f /data/DBpedia_3_class_full/plan_5_int/test_pred.csv -a Plan5int -o results/ --objective objective.py


#partially unseen PlanRGCN
python3 /PlanRGCN/scripts/post_predict_unseenPartially.py -s /data/DBpedia_3_class_full -t 3 -f /data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv -a PlanRCN -o results/ --objective objective.py

#partially unseen NN
#python3 /PlanRGCN/scripts/post_predict_unseenPartially.py -s /data/DBpedia_3_class_full -t 3 -f nn_results/nn_pred.csv -a NN -o nn_results/ --objective objective.py

python3 /PlanRGCN/scripts/baseline_snap.py -s /data/DBpedia_3_class_full -t 3 -f /data/DBpedia_3_class_full/nn/k25/nn_test_pred.csv -o nn_results/ --objective objective.py
python3 /PlanRGCN/scripts/post_predict.py -s /data/DBpedia_3_class_full -t 3 -f nn_results/nn_pred.csv -o nn_results/ --objective objective.py
python3 /PlanRGCN/scripts/post_predict_unseen.py -s /data/DBpedia_3_class_full -t 3 -f nn_results/nn_pred.csv -a NN -o nn_results/ --objective objective.py


exit
python3 /PlanRGCN/scripts/baseline_snap.py -s /data/DBpedia_3_class_full -t 3 -f /data/DBpedia_3_class_full/svm/test_pred_reg.csv -o svm_results/ --objective objective.py
python3 /PlanRGCN/scripts/post_predict.py -s /data/DBpedia_3_class_full -t 3 -f svm_results/svm_pred.csv -o svm_results/ --objective objective.py
python3 /PlanRGCN/scripts/post_predict_unseen.py -s /data/DBpedia_3_class_full -t 3 -f svm_results/svm_pred.csv -a SVM -o svm_results/ --objective objective.py