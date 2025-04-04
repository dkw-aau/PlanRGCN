from qpp_new.data_prep import NNDataPrepper, OriginalDataPrepper, SVMDataPrepper
from argparse import ArgumentParser
import os
from qpp_new.models.svm import SVMTrainer, SVMTrainerNoPreprocess
from qpp_new.models.NN import NNTrainer
from qpp_new.models.twostepsvm import TwoStepSVM
import pickle as pcl

if __name__ == "__main__":
    #python3 qpp_new.trainer nn --data-dir /data/DBpedia2016_0_1_10_path_weight_loss --results-dir /data/DBpedia2016_0_1_10_path_weight_loss --k 25
    
    parser = ArgumentParser("QPP baseline trainer")
    parser.add_argument("task", choices=["svm", "nn", "two-step"])
    parser.add_argument(
        "--data-dir", help="Path to the directory containing dataset files"
    )
    parser.add_argument(
        "--pred_dir", help="Path to the directory containing other queries dataset files. Needs to contain test_alg.tsv and test_ged.csv"
    )
    parser.add_argument(
        "--results-dir",
        default="/qpp/dataset/DBpedia_2016_12k_sample/results",
        help="Path to the results directory",
    )
    parser.add_argument(
        "--new_test_path",
        default=None,
        help="Path to the test new path",
    )
    parser.add_argument("--k", type=int, help="Value of k", default=25)

    args = parser.parse_args()
    data_dir = args.data_dir #or "/qpp/dataset/DBpedia_2016_12k_sample"
    pred_dir = args.pred_dir
    res_dir = args.results_dir
    K = args.k
    # res_dir = "/qpp/dataset/DBpedia_2016_12k_sample/results"
    match args.task:
        case "svm":
            prepper = SVMDataPrepper(
                train_algebra_path=os.path.join(data_dir, "baseline","train_alg.tsv"),
                val_algebra_path=os.path.join(data_dir, "baseline","val_alg.tsv"),
                test_algebra_path=os.path.join(data_dir, "baseline", "test_alg.tsv"),
                train_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/train_ged.csv"),
                val_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/val_ged.csv"),
                test_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/test_ged.csv"),
            )
            train, val, test = prepper.prepare()
            
            resultDir = f"{res_dir}/svm/"
            os.system(f"mkdir -p {resultDir}")
            trainer = SVMTrainer(train, val, test, resultDir)
            trainer.trainer()
            with open(f"{resultDir}svmtrainer.pickle",'wb') as f:
                pcl.dump(trainer, f)
            
        case "nn":
            prepper = NNDataPrepper(
                train_algebra_path=os.path.join(data_dir, "baseline", "train_alg.tsv"),
                val_algebra_path=os.path.join(data_dir, "baseline", "val_alg.tsv"),
                test_algebra_path=os.path.join(data_dir, "baseline", "test_alg.tsv"),
                train_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/train_ged.csv"),
                val_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/val_ged.csv"),
                test_ged_path=os.path.join(data_dir, "baseline", f"knn{K}/test_ged.csv"),
                filter_join_data_path=os.path.join(data_dir, "baseline", f"extra"),
            )
            train, val, test = prepper.prepare()
            resultDir = f"{res_dir}/nn/k{K}/"
            os.system(f"mkdir -p {resultDir}")
            trainer = NNTrainer(train, val, test, resultDir)
            trainer.trainer()
            with open(f"{resultDir}nntrainer.pickle",'wb') as f:
                pcl.dump(trainer, f)
        case "two-step":
            prepper = SVMDataPrepper(
                train_algebra_path=os.path.join(data_dir, "train_sampled.tsv"),
                val_algebra_path=os.path.join(data_dir, "val_sampled.tsv"),
                test_algebra_path=os.path.join(data_dir, "test_sampled.tsv"),
                train_ged_path=os.path.join(data_dir, f"knn{K}/train_ged.csv"),
                val_ged_path=os.path.join(data_dir, f"knn{K}/val_ged.csv"),
                test_ged_path=os.path.join(data_dir, f"knn{K}/test_ged.csv"),
            )
            train, val, test = prepper.prepare()
            resultDir = f"{res_dir}/two_svm"

            predictor = TwoStepSVM(train, val, test, resultDir, prepper)
            predictor.trainer()
            # cluster = XMeansCluster(np.reshape(train["time"].to_numpy(), (-1, 1)), 5)
            # train = cluster.assign_class(train, time_col="time")
            # print(train["class_labels"].unique())
        case "original-svm":
            d = OriginalDataPrepper()
            train, val, test = d.prepare()
            resultDir = f"{res_dir}/original_svm/"
            os.system(f"mkdir -p {resultDir}")
            trainer = SVMTrainerNoPreprocess(train, val, test, resultDir)
            trainer.trainer()
