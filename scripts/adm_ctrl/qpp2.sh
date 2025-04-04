wikidata_qpp2 (){
    #config_path=/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
    CPUS="10"
    dbpath=/data/abiram/wdbench/virtuoso_dabase
    imp_path=/data/abiram/wdbench/import
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    #-v $config_path:/database/virtuoso.ini
    docker run -m 64g --rm --name wdbench_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v imp_path:/import --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}

dbpedia_qpp2 (){
    #config_path=/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
    CPUS="10"
    dbpath=/data/abiram/DBpedia2016/virtuoso-db-new
    imp_path=/data/abiram/DBpedia2016/import
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    #-v $config_path:/database/virtuoso.ini
    docker run -m 64g --rm --name dbpedia_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v imp_path:/import --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}

 # dont need this anymore
STARTIME=60


#General Configurations
DATAPATH='/data/abiram/data_qpp'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/workload1'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
SEED=42
STARTIME=60

#Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
PRED_COL='planrgcn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn_44'
WORKLOAD_FILE=$BASEOUTPUT_FOLDER/'workload.pcl'

adm_ctrl () {
  docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $1 \
  -p $2 \
  -t $3 \
  -o $4 -r $5 \
  -u $6 \
  -i $7 \
  --seed $8 \
  --interval 2 \
  --workload_file ${9}
}

dbpedia_qpp2

sleep $STARTIME
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL $OUTPUT_FOLDER $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED $WORKLOAD_FILE

exit

echo "Starting DBpedia PlanRGCN afster $SECONDS"
docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2

echo "FINISHED DBpedia PlanRGCN afster $SECONDS"
docker stop dbpedia_virt && dbpedia_qpp2
sleep $STARTIME

START=$SECONDS
PREDICTION_FILE='/data/DBpedia_3_class_full/nn/test_pred.csv'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/nn_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
echo "Starting DBpedia NN afster $SECONDS"
docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2 \
  -l no

echo "FINISHED DBpedia NN after $SECONDS"
docker stop dbpedia_virt && dbpedia_qpp2
sleep $STARTIME

START=$SECONDS
PREDICTION_FILE='/data/DBpedia_3_class_full/svm/test_pred_cls.csv'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/svm_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
echo "Starting DBpedia SVM afster $SECONDS"
docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2 \
  -l no

echo "Finishing DBpedia SVM afster $SECONDS"
docker stop dbpedia_virt && wikidata_qpp2

echo "WIKIDATA"
sleep $STARTIME

DATAPATH='/data/abiram/data_qpp'
PREDICTION_FILE='/data/wikidata_3_class_full/planRGCN_no_pred_co/test_pred.csv'
SPLIT_FILE='/data/wikidata_3_class_full/test_sampled.tsv'
PRED_COL='planrgcn_prediction'
OUTPUT_FOLDER='/data/wikidata_3_class_full/admission_control/planrgcn_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
echo "Starting Wikidata PlanRGCN after $SECONDS"
docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2

echo "FINISHED wikidata PlanRGCN after $SECONDS"
docker stop wdbench_virt && wikidata_qpp2
sleep $STARTIME

START=$SECONDS
PREDICTION_FILE='/data/wikidata_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER='/data/wikidata_3_class_full/admission_control/nn_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
echo "Starting Wikidata NN after $SECONDS"
docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2 \
  -l no

echo "FINISHED Wikidata NN after $SECONDS"
docker stop wdbench_virt && wikidata_qpp2
sleep $STARTIME

START=$SECONDS
PREDICTION_FILE='/data/wikidata_3_class_full/baseline/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER='/data/wikidata_3_class_full/admission_control/svm_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
echo "Starting Wikidata SVM after $SECONDS"
docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2 \
  -l no

echo "Finishing Wikidata SVM afster $SECONDS"

exit













exit
#already run correctly presumably, Make analysis to be sure.
DATAPATH='/data/abiram/data_qpp'
PREDICTION_FILE='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
PRED_COL='planrgcn_prediction'
OUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/planrgcn_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10

docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2