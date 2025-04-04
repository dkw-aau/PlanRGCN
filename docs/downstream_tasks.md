## Load Balancing
Start the RDF store with the Virtuoso RDF store init files in virt_feat_conf folder.
To run the experiment execute the following:
```
(cd $LB/load_balance_SVM_44_10_workers && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
```

## Execution Control
```
#Cross experiment setting
WORKLOADID="1"
WORKERS=10
STARTIME=120
ARRIVAL_RATE=44
ENDPOINT=
DATAPATH=
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
QUANTILE95=
#General Configurations - DBpedia
CONTAINER_NAME='dbpedia_virt'
VIRT_CONFIG=
dbpath=
SPLIT_FILE=
BASEOUTPUT_FOLDER=


db_start
sleep $STARTIME

PREDICTION_FILE=
PRED_COL="planrgcn_prediction"
OUTPUT_FOLDER=$BASEOUTPUT_FOLDER'/planrgcn'
WORKLOADFILE="${OUTPUT_FOLDER}/workload.pck"
sleep $STARTIME
echo "DBPEDIA PlanRGCN start ${SECONDS}"
adm_ctrl $SPLIT_FILE $PREDICTION_FILE $PRED_COL "$OUTPUT_FOLDER" $ARRIVAL_RATE $ENDPOINT $ACCEPT_WORKERS $REJECT_WORKERS $SEED "yes"
echo "DBPEDIA PlanRGCN END ${SECONDS}"
docker stop $CONTAINER_NAME
```