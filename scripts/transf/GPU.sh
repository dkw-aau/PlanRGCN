

transfer_wikidataV2_qpp2(){
BASE=/data/abiram/data_qpp

params=(--exclude 'baseline' --exclude "data_splitter.pickle" --exclude "geddbtran.sh" --exclude "plan01" --exclude "queryStat.json" --exclude "resampled" --exclude "knn25")
echo "${params[@]}"
rsync -aWP "${params[@]}" qpp2:"${BASE}/wikidataV2" /data/
}


transfer_dbpedia_qpp(){
BASE=/srv/data/abiram/data_qpp

params=(--exclude 'dist_time_20_cpu.log' --exclude 'feature.log' --exclude 'comb_time_20_cpu.log' --exclude 'baseline' --exclude 'admission_control*' --exclude 'ray_save' --exclude 'distances' --exclude 'planWOPredCo40Epochs' --exclude 'planRGCNWpredCo' --exclude 'load_balance*' --exclude 'combinations' --exclude "data_splitter.pickle" --exclude "geddbtran.sh" --exclude "plan01" --exclude "queryStat.json" --exclude "resampled" --exclude "knn25")
echo "${params[@]}"
rsync -aWP "${params[@]}" qpp:"${BASE}/DBpedia_3_class_full" /data/
}