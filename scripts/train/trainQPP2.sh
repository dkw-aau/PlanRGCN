echo "Training model in QPP2"
export QG_JAR=/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar
EXP_NAME="Wikidata Training"
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidata_3_class_full/plan01_n_gq
BASEFOLDERNAME=wikidata_3_class_full
LAYER1=4096
LAYER2=2048
NCLASSES=3
mkdir -p $EXP
echo """
import numpy as np
def cls_func(lat):
    vec = np.zeros(3)
    if lat <= 1:
        vec[0] = 1
    elif (1 < lat) and (lat <= 10):
        vec[1] = 1
    elif (10 < lat):
        vec[2] = 1
    return vec
thresholds = [0, 1, 10, 900]
n_classes = 3
"""> "$EXP"/objective.py
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py $BASEFOLDERNAME $EXP --feat_path $FEAT --use_pred_co no --layer1_size $LAYER1 --layer2_size $LAYER2 --class_path "$EXP"/objective.py
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n $NCLASSES -o "$EXP" --l1 $LAYER1 --l2 $LAYER2
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt

exit



echo "Training model in QPP2"
export QG_JAR=/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar
EXP_NAME="DBpedia Training"
FEAT=/data/metaKGStat/dbpedia
EXP=/data/DBpediaV2/plan01
BASEFOLDERNAME=DBpediaV2
LAYER1=4096
LAYER2=1024
NCLASSES=3
mkdir -p $EXP
echo """
import numpy as np
def cls_func(lat):
    vec = np.zeros(3)
    if lat <= 0.0352054542551437:
        vec[0] = 1
    elif (0.0352054542551437 < lat) and (lat <= 50.830727839345734):
        vec[1] = 1
    elif (50.830727839345734 < lat):
        vec[2] = 1
    return vec
thresholds = [0, 0.0352054542551437, 50.830727839345734, 900]
n_classes = 3
"""> "$EXP"/objective.py
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py $BASEFOLDERNAME $EXP --feat_path $FEAT --use_pred_co no --layer1_size $LAYER1 --layer2_size $LAYER2 --class_path "$EXP"/objective.py
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n $NCLASSES -o "$EXP" --l1 $LAYER1 --l2 $LAYER2
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
exit


echo "Training model in QPP2"
export QG_JAR=/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar
EXP_NAME="Wikidata Training"
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidataV2/plan01
BASEFOLDERNAME=wikidataV2
LAYER1=8192
LAYER2=4096
NCLASSES=3
mkdir -p $EXP
echo """
import numpy as np
def cls_func(lat):
    vec = np.zeros(3)
    if lat <= 0.003958:
        vec[0] = 1
    elif (0.003958 < lat) and (lat <= 1.294):
        vec[1] = 1
    elif (1.294 < lat):
        vec[2] = 1
    return vec
thresholds = [0, 0.003958, 1.294, 900]
n_classes = 3
"""> "$EXP"/objective.py
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py $BASEFOLDERNAME $EXP --feat_path $FEAT --use_pred_co no --layer1_size $LAYER1 --layer2_size $LAYER2 --class_path "$EXP"/objective.py
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n $NCLASSES -o "$EXP" --l1 $LAYER1 --l2 $LAYER2
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt

exit

EXP_NAME="Wikidata Training"
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidata_3_class_full/plan_something
BASEFOLDERNAME=wikidata_3_class_full
LAYER1=8192
LAYER2=4096
NCLASSES=3
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py $BASEFOLDERNAME $EXP --feat_path $FEAT --use_pred_co no --layer1_size $LAYER1 --layer2_size $LAYER2
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n $NCLASSES -o "$EXP" --l1 $LAYER1 --l2 $LAYER2
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt






exit

## Temp code for making new splits
F=/data/wikidataV2
OF=/data/wikidata_3_class_full
mkdir $F
cd $F
cp $OF/all.tsv $F
cp $OF/test_sampled.tsv $F
cp $OF/train_sampled.tsv $F
cp $OF/val_sampled.tsv $F