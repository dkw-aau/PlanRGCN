

: 'basedir=/qpp/dataset/DBpedia2016_sample_0_1_10
dist_dir=/SPARQLBench/dbpedia2015_16/ged_dir_ordered2015_2016
python3 /qpp/qpp_new/qpp_new/feature_combiner.py $basedir $dist_dir
'


dist_dir=/SPARQLBench/dbpedia2015_16/ged_dir_ordered2015_2016
basedir=/qpp/dataset
# Define the list of folder values
configs=("DBpedia2016_sample_0_1_10_weight_loss" "DBpedia2016_sample_0_1_10_aug")
for config in "${configs[@]}"; do
    # Split the config into basedir and snap_lat2onehot_binary
    IFS=' ' read -r folder <<< $config
    complete_path=$basedir/$folder
    python3 /qpp/qpp_new/qpp_new/feature_combiner.py $complete_path $dist_dir
done
