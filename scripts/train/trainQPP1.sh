

exit
EXP_NAME="Wikidata Training"
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidata_3_class_full/plan_l18192_l24096_no_pred_co
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py wikidata_3_class_full $EXP --feat_path $FEAT --use_pred_co no --layer1_size 8192 --layer2_size 4096
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 8192 --l2 4096
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt


exit
#Running this concurrently with upper experiment.
EXP_NAME="DBpedia Training"
FEAT=/data/metaKGStat/dbpedia
EXP=/data/DBpedia_3_class_full/plan_l18192_l24096_no_pred_co
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --use_pred_co no --layer1_size 4096 --layer2_size 1024
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 1024
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt


exit
FEAT=/data/metaKGStat/dbpedia
EXP=/data/DBpedia_3_class_full/graphsage
mkdir -p $EXP
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --use_pred_co no --conv_type "SAGE"
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 4096 --conv_type "SAGE"


exit # temp
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidata_3_class_full/graphsage
mkdir -p $EXP
python3 /PlanRGCN/scripts/train/ray_run.py wikidata_3_class_full $EXP --feat_path $FEAT --use_pred_co no --conv_type "SAGE"
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 4096 --conv_type "SAGE"



exit

# FROM GPU Server
echo with custom intervals
EXP=/data/DBpedia_3_class_full/plan_5_int_5min
mkdir -p $EXP
echo """
import numpy as np
def cls_func(lat):
    vec = np.zeros(5)
    if lat < 0.004:
        vec[0] = 1
    elif (0.004 < lat) and (lat <= 1):
        vec[1] = 1
    elif (1 < lat) and (lat <= 10):
        vec[2] = 1
    elif (10 < lat) and (lat <= 300):
        vec[3] = 1
    elif 300 < lat:
        vec[4] = 1
    return vec
n_classes = 5
"""> "$EXP"/objective.py
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --class_path "$EXP"/objective.py


exit
echo Code from gpu server

echo DBpedia train
FEAT=/data/metaKGStat/dbpedia
EXP=/data/DBpedia_3_class_full/planWOPredCo40Epochs
mkdir -p $EXP
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --use_pred_co no
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 4096


echo Wikidata Training
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidata_3_class_full/planWOPredCo40Epochs
mkdir -p $EXP
python3 /PlanRGCN/scripts/train/ray_run.py wikidata_3_class_full $EXP --feat_path $FEAT --use_pred_co no
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 4096