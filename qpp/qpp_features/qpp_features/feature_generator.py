import argparse
from qpp_features.ged_calculator import main as gmain
import os

if __name__ == "__main__":
    #python3 -m qpp_features.ged_calculator /data/DBpedia2016h_weight_loss/all.tsv /data/dbpedia_dist2 
    #python3 -m qpp_features.ged_calculator /data/DBpedia2016h_weight_loss/all.tsv /data/dbpedia_dist2 -t dist_calc
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description="Script redirector.")

    # Add input and output file arguments
    parser.add_argument("query_log_file", type=str, help="Input query log file")
    parser.add_argument("output_folder", type=str, help="Output distance matrix folder")
    parser.add_argument("-t", type=str, help="task", default='combinations', choices=["combinations", "dist_calc", "alg_feat"])
    parser.add_argument("-cpus", type=int, help="cpus", default=20)


    # Parse the command line arguments
    args = parser.parse_args()
    query_log_file = args.query_log_file
    output_folder = args.output_folder
    task = args.t
    if task in ["combinations", "dist_calc"]:
        gmain(query_log_file, output_folder, task, cpus=args.cpus)
    elif task == "alg_feat":
        #python3 -m qpp_features.feature_generator /data/DBpedia2016_0_1_10_path_weight_loss/all.tsv /data/DBpedia2016_0_1_10_path_weight_loss -t alg_feat
        #python3 -m qpp_features.feature_generator /data/wikidata_0_1_10_v2_path_weight_loss/all.tsv /data/wikidata_0_1_10_v2_path_weight_loss -t alg_feat
        output_file2 = os.path.join(output_folder,'baseline','extra')
        jarfile = "/PlanRGCN/qpp/jars/sparql-query2vec-0.0.1.jar"
        if not os.path.exists(jarfile):
            os.system("mvn package -f /qpp/qpp_features/sparql-query2vec/pom.xml")
        for i in [("train_alg.tsv","train_sampled.tsv"), ("val_alg.tsv","val_sampled.tsv"),("test_alg.tsv","test_sampled.tsv")]:
            output_file1 = os.path.join(output_folder,'baseline',i[0])
            input_file1 = os.path.join(output_folder, i[1])
            os.system(f"java -jar {jarfile} algebra-features {input_file1} {output_file1}")
        os.system(f"java -jar {jarfile} extra {query_log_file} {output_file2}")
