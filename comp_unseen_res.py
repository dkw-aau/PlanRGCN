import os
from pathlib import Path

os.environ['QG_JAR'] = '/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR'] = '/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'

from trainer.new_post_predict import QPPResultProcessor


def qppDBpediaNewqs(exp_type="completly_unseen"):
    base = '/data/DBpedia_3_class_full/newUnseenQs4'
    dbpedia_plan_path = os.path.join(base, 'plan_inference.csv')
    base_path = '/data/DBpedia_3_class_full'

    val_sampled_file = os.path.join(base_path, 'val_sampled.tsv')
    train_sampled_file = os.path.join(base_path, 'train_sampled.tsv')

    p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py',
                           dataset="DBpedia",
                           exp_type=exp_type,
                           test_sampled_file=os.path.join(base, "queries.tsv"),
                           val_sampled_file=val_sampled_file,
                           train_sampled_file=train_sampled_file)

    dbpedia_nn_path = os.path.join(base, 'nn_prediction.csv')
    dbpedia_svm_path = os.path.join(base, 'svm_pred.csv')

    p.evaluate_dataset(path_to_pred=dbpedia_plan_path,
                       sep=',',
                       ground_truth_col='time_cls',
                       pred_col='pred',
                       id_col='id',
                       approach_name="P",
                       reg_to_cls=False)

    p.evaluate_dataset(path_to_pred=dbpedia_nn_path,
                       sep=',',
                       ground_truth_col='time_cls',
                       pred_col='nn_prediction',
                       id_col='id',
                       approach_name="NN", reg_to_cls=True)
    p.evaluate_dataset(path_to_pred=dbpedia_svm_path,
                       sep=',',
                       ground_truth_col='time_cls',
                       pred_col='svm_prediction',
                       id_col='id',
                       approach_name="SVM", reg_to_cls=True)
    print(p.process_results(add_symbol=''))
    print(p.true_counts)


qppDBpediaNewqs()
