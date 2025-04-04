transfer_dbpedia_gpu(){
BASE=/home/tq74iz/data_qpp

params=(--exclude 'dist_time_20_cpu.log' --exclude 'feature.log' --exclude 'comb_time_20_cpu.log' --exclude 'baseline' --exclude 'admission_control*' --exclude 'distances' --exclude 'planWOPredCo40Epochs' --exclude 'planRGCNWpredCo' --exclude 'load_balance*' --exclude 'combinations' )
echo "${params[@]}"
rsync -aWP "${params[@]}" gpu5:"${BASE}/DBpedia_3_class_full" /data/
}

DBpediaV2_gpu(){
BASE=/home/tq74iz/data_qpp/DBpediaV2

params=(--exclude 'dist_time_20_cpu.log' --exclude 'feature.log' --exclude 'comb_time_20_cpu.log' --exclude 'baseline' --exclude 'admission_control*' --exclude 'distances' --exclude 'planWOPredCo40Epochs' --exclude 'planRGCNWpredCo' --exclude 'load_balance*' --exclude 'combinations' )
echo "${params[@]}"
rsync -aWP "${params[@]}" gpu5:"${BASE}" /data/
}

wikidataV2_gpu(){
BASE=/home/tq74iz/data_qpp/wikidataV2

params=(--exclude 'dist_time_20_cpu.log' --exclude 'feature.log' --exclude 'comb_time_20_cpu.log' --exclude 'baseline' --exclude 'admission_control*' --exclude 'distances' --exclude 'planWOPredCo40Epochs' --exclude 'planRGCNWpredCo' --exclude 'load_balance*' --exclude 'combinations' )
echo "${params[@]}"
rsync -aWP "${params[@]}" gpu5:"${BASE}" /data/
}


/home/tq74iz/data_qpp/DBpediaV2