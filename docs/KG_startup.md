## Starting Virtuoso Instances
The data should be loaded in.

For DBpedia,
```
CONTAINER_NAME='dbpedia_virt'
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_dbpedia_load_balance.ini
dbpath=/srv/data/abiram/dbpediaKG/virtuoso-db-new2/virtuoso-db-new
```


For Wikidata,
```
CONTAINER_NAME=
VIRT_CONFIG=
dbpath=/
```

For actually running the RDF store:
```
db_start (){
    CPUS="10"
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    docker run -m 64g --rm --name $CONTAINER_NAME -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v $VIRT_CONFIG:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}
db_start
```