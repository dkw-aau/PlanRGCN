Run the following command in the super directory where this is cloned
```
DATA_PATH=$(pwd)/PlanRGCN/data
docker run --name all_final -it -v $(pwd)/PlanRGCN:/PlanRGCN -v $(pwd)/PlanRGCN/qpp:/qpp -v $DATA_PATH:/data --shm-size=12gb ubuntu:22.04
```
when in the container, run:
```
(cd PlanRGCN && bash scripts/setup.sh)
(cd qpp && bash scripts/ini.sh)
```
The query logs with train, val, test is provided in wikidata_3_full and DBpedia_3_interval_full in data folder.
Otherwise:
Then untar the query log split and files:
```
tar -xvf /PlanRGCNdata/qpp_datasets.tar.gz
```
The collected MetaKG statistics are not included due to the file size limit on Github.
Nevetherless, the Virtuoso Endpoint creation is specified in /Datasets/KG