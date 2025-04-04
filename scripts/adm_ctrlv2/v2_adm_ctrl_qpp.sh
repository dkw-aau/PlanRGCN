
#Cross experiment setting
WORKLOADID="1"
WORKERS=10
STARTIME=120
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.23:8891/sparql
DATAPATH="/srv/data/abiram/data_qpp"
SEED=21
ACCEPT_WORKERS=7
REJECT_WORKERS=3

#Database
db_start (){
    CPUS="10"
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    docker run -m 64g --rm --name $CONTAINER_NAME -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v $VIRT_CONFIG:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}

adm_ctrl () {
  docker run --rm --name adm_ctrl -v $DATAPATH:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  --task "v2" \
  -f "$1" \
  -p "$2" \
  -t "$3" \
  -o "$4" -r "$5" \
  -u "$6" \
  --acceptWorkers "$7" \
  --rejectWorkers "$8" \
  --seed "$9" \
  -l "${10}" \
  --interval 2 \
  --rejectTimeout "$QUANTILE95" \
  --workload_file "${WORKLOADFILE}"
}

#### DBpedia

QUANTILE95=50.83
QUANTILE95=51
#General Configurations - DBpedia
CONTAINER_NAME='dbpedia_virt'
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_dbpedia_load_balance.ini
dbpath=/srv/data/abiram/dbpediaKG/virtuoso-db-new2/virtuoso-db-new
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER="/data/DBpedia_3_class_full/admission_controlV2/workload${WORKLOADID}"


db_start
sleep $STARTIME
# NN Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA NN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "no"
echo "DBPEDIA NN END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

# SVM Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/svm'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA SVM start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "no"
echo "DBPEDIA SVM END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

PREDICTION_FILE='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
PRED_COL="planrgcn_prediction"
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA PlanRGCN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "yes"
echo "DBPEDIA PlanRGCN END ${SECONDS}"
docker stop $CONTAINER_NAME

#### Wikidata

QUANTILE95=1.294
QUANTILE95=1
#General Configurations - Wikidata
EXP_NAME="wikidata"
dbpath=/srv/data/abiram/wdbench/virtuoso_dabase
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
CONTAINER_NAME="wikidata_virt"
SPLIT_FILE='/data/wikidata_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER="/data/wikidata_3_class_full/admission_controlV2/workload${WORKLOADID}"

db_start
sleep $STARTIME
# NN Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "${EXP_NAME} NN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "no"
echo "${EXP_NAME} NN END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

# SVM Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/baseline/svm/test_pred_cls.csv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/svm'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "${EXP_NAME} SVM start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "no"
echo "${EXP_NAME} SVM END ${SECONDS}"
docker stop $CONTAINER_NAME && db_start

# PlanRGCN Experiment related configurations
PREDICTION_FILE='/data/wikidata_3_class_full/planRGCN_no_pred_co/test_pred.csv'
PRED_COL="planrgcn_prediction"
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "${EXP_NAME} PlanRGCN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "yes"
echo "${EXP_NAME} PlanRGCN END ${SECONDS}"
docker stop $CONTAINER_NAME