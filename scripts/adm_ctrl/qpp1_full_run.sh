## Notes 4 and 3 are the same runs, just that the lsq option on baselines were wrong in 3

#Cross experiment setting
WORKLOADID="4"
WORKERS=10
STARTIME=120
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.23:8891/sparql
DATAPATH="/srv/data/abiram/data_qpp"
SEED=21

#DBpedia QPP1 server
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
  --interval 2
}

#General Configurations - DBpedia
CONTAINER_NAME='dbpedia_virt'
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_dbpedia_load_balance.ini
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER="/data/DBpedia_3_class_full/admission_control/workload${WORKLOADID}_${SEED}"
dbpath=/srv/data/abiram/dbpediaKG/virtuoso-db-new2/virtuoso-db-new

db_start
sleep $STARTIME
# NN Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
sleep $STARTIME
echo "DBPEDIA NN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"
echo "DBPEDIA NN END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

# SVM Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/svm'
sleep $STARTIME
echo "DBPEDIA SVM start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"
echo "DBPEDIA SVM END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start


# PlanRGCN Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
PRED_COL="planrgcn_prediction"
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'
sleep $STARTIME
echo "DBPEDIA PlanRGCN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "yes"
echo "DBPEDIA PlanRGCN END ${SECONDS}"
docker stop $CONTAINER_NAME

### Wikidata experiments
#General Configurations - Wikidata
EXP_NAME="wikidata"
dbpath=/srv/data/abiram/wdbench/virtuoso_dabase
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
CONTAINER_NAME="wikidata_virt"
SPLIT_FILE='/data/wikidata_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER="/data/wikidata_3_class_full/admission_control/workload${WORKLOADID}_${SEED}"

db_start
sleep $STARTIME
# NN Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
sleep $STARTIME
echo "${EXP_NAME} NN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"
echo "${EXP_NAME} NN END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

# SVM Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/baseline/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/svm'
sleep $STARTIME
echo "${EXP_NAME} SVM start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"
echo "${EXP_NAME} SVM END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start


# PlanRGCN Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/planRGCN_no_pred_co/test_pred.csv'
PRED_COL="planrgcn_prediction"
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'
sleep $STARTIME
echo "${EXP_NAME} PlanRGCN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "yes"
echo "${EXP_NAME} PlanRGCN END ${SECONDS}"
docker stop $CONTAINER_NAME