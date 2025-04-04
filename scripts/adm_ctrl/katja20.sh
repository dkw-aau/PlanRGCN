#DBpedia QPP1 server
db_start (){
    CPUS="10"
    dbpath=/srv/data/tq74iz/DBpedia/database
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    #-v $config_path:/database/virtuoso.ini
    docker run -m 64g --rm --name dbpedia_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}


CONTAINER_NAME='dbpedia_virt'
#General Configurations
DATAPATH='/srv/data/tq74iz/data_qpp'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
BASEOUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/workload2_'${SEED}
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.30:8891/sparql
WORKERS=10
SEED=21
STARTIME=120


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


# NN Experiment related configurations
PREDICTION_FILE='/data/DBpedia_3_class_full/nn/test_pred.csv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/nn'
db_start
sleep $STARTIME
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $WORKERS $SEED "no"
docker stop $CONTAINER_NAME

