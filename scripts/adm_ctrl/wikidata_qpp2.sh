db_start (){
    #config_path=/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
    CPUS="10"
    dbpath=/data/abiram/wdbench/virtuoso_dabase
    imp_path=/data/abiram/wdbench/import
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    #-v $config_path:/database/virtuoso.ini
    docker run -m 64g --rm --name wdbench_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v imp_path:/import --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}

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
  -l $9 \
  --interval 2
}


CONTAINER_NAME='wdbench_virt'
#General Configurations
SEED=21
DATAPATH='/data/abiram/data_qpp'
SPLIT_FILE='/data/wikidata_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER='/data/wikidata_3_class_full/admission_control/workload1_'${SEED}
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10
STARTIME=120



# NN Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
db_start
sleep $STARTIME
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL $OUTPUT_FOLDER $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"

# SVM Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/baseline/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/svm'
docker stop $CONTAINER_NAME && db_start
sleep $STARTIME
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL $OUTPUT_FOLDER $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"

# PlanRGCN Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/planRGCN_no_pred_co/test_pred.csv'
PRED_COL='planrgcn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'

docker stop $CONTAINER_NAME && db_start
sleep $STARTIME
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL $OUTPUT_FOLDER $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "yes"