# PlanRGCN

## Prerequisites
Docker should be installed

A folder containing the dataset, i.e., the query logs divided into training, validation, and test sets should be available at $DATA

For reproducing this work, please download our query logs at [this link](https://drive.google.com/drive/folders/1mYb6MmhRDFXEN5XmY9S0-Yt3ToJp9CYC?usp=sharing).



## Setup
Please build the docker image by
```
docker build -t plan:latest . -f Dockerfile
```
with GPU support:
```
docker build -t plan:latest . -f DockerfileGPU
```
To run the different command, a container can be started in interactive mode:
```
#if GPU support
GPU='--gpus all'
#if no GPU

docker run --rm -it -v $DATA:/data -v $PWD/PlanRGCN:/PlanRGCN $GPU --name planrgcn plan:latest
```

The query logs with train, val, test is provided in wikidata_3_full and DBpedia_3_interval_full in data folder.
Otherwise:
Then untar the query log split and files:
```
tar -xvf /PlanRGCNdata/qpp_datasets.tar.gz
```

The collected MetaKG statistics are not included due to the file size limit on Github.
Nevetherless, the Virtuoso Endpoint creation is specified in /Datasets/KG

## Feature Extraction
See [feature extraction documentation](docs/MetaKGStat.md)

## Model Training
First, we identify the best model through hyper-parameter searching:
```

FEAT={PATH to extracted MetaKG stats}
DATESTRING={$DATE}
EXP=/data/{$DATASET}/plan${DATESTRING}
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/hyper_log.txt
python3 /PlanRGCN/scripts/train/ray_hyperparam.py {$DATASET} $EXP --feat_path $FEAT --use_pred_co no --class_path {$EXP}/objective.py
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/hyper_log.txt
(cd $EXP && tar --use-compress-program="pigz --best --recursive" -cf ray_save.tar.gz ray_save)
```
Note, that after identifying the set of best configuration, later model traning should just use those values instead of performing an expensive hyperparemeter search.

For model prediction/results:
```
L1={first layer size}
L2={second layer size}
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n {$NUMBER_OF_CLASSES} -o "$EXP" --l1 $L1 --l2 $L2
```

## Processing results
```
SPLIT_DIR=/data/wikidata_3_class_full
TIMECLS=3
EXP_NAME=plan_l18192_l24096_no_pred_co
PRED_FILE="$SPLIT_DIR"/"$EXP_NAME"/test_pred.csv
APPROACH="PlanRGCN"
OUTPUTFOLDER="$SPLIT_DIR"/"$EXP_NAME"/results
python3 /PlanRGCN/scripts/post_predict.py -s $SPLIT_DIR -t $TIMECLS -f $PRED_FILE -a $APPROACH -o $OUTPUTFOLDER

```
Prediction quality is analyzed in the notebooks in the notebook folder.


## Live predictions through terminal
To get prediction on user queries, the following command can be run:

```
MODEL_PATH=
python3 /PlanRGCN/online_predictor.py \
     --prep_path $MODEL_PATH/prepper.pcl\
     --model_path $MODEL_PATH/best_model.pt\
     --config_path $MODEL_PATH/model_config.json\
     --gpu yes
```
NOTE: the SPARQL query should only on a single line, e.g., no newline characters.

To test the model, we provide our trained model [here](https://drive.google.com/drive/folders/1mYb6MmhRDFXEN5XmY9S0-Yt3ToJp9CYC?usp=sharing).
Download the tar achieve and extract it /data folder in container.

## Downstream tasks
See [Downstream tasks documentation](docs/downstream_tasks.md)

## Baseline Methods
See [documentation](docs/baselines.md)


