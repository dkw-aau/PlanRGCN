#Cross experiment setting
WORKERS=10
STARTIME=120
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.23:8891/sparql
DATAPATH="/srv/data/abiram/data_qpp"
SEED=21

#QPP1 datebasd
db_start (){
    CPUS="10"
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    docker run -m 64g --rm --name $CONTAINER_NAME -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v $VIRT_CONFIG:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}


adm_ctrl () {
  docker run --rm --name adm_ctrl -v $DATAPATH:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $1 \
  -p $2 \
  -t $3 \
  -o $4 -r $5 \
  -u $6 \
  -i $7 \
  --seed $8 \
  -l $9 \
  --workload_file ${10} \
  --interval 2
}

#General Configurations - DBpedia
CONTAINER_NAME='dbpedia_virt'
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_dbpedia_load_balance.ini
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER="/data/DBpedia_3_class_full/admission_control/workload5_lb"
BASEOUTPUT_FOLDER="/data/DBpedia_3_class_full/admission_control/workload6"
dbpath=/srv/data/abiram/dbpediaKG/virtuoso-db-new2/virtuoso-db-new

db_start
sleep $STARTIME
# NN Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA NN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no" $WORKLOADFILE
echo "DBPEDIA NN END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

# SVM Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/svm'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA SVM start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no" $WORKLOADFILE
echo "DBPEDIA SVM END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start


# PlanRGCN Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
PRED_COL="planrgcn_prediction"
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA PlanRGCN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no" $WORKLOADFILE
echo "DBPEDIA PlanRGCN END ${SECONDS}"
docker stop $CONTAINER_NAME
