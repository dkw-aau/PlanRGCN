# Virtuoso Triple Store Setup
Modified version of https://gist.github.com/kuzeko/5d53f9800a4b6d45006f0f9dc322ed07

## Setup
```
docker pull openlink/virtuoso-opensource-7:latest
mkdir -p virtuoso-db #these cannot be run on server but has to be run on a container instead.
cp virtuoso.ini virtuoso-db/virtuoso.ini #these cannot be run on server but has to be run on a container instead.
//mkdir -p import #these cannot be run on server but has to be run on a container instead.
```
Comamnd to run container:
```
docker run \
    --name vos \
    -it \
    --tty \
    --rm \
    --env DBA_PASSWORD=dba \
    --env DAV_PASSWORD=dba \
    --publish 1111:1111 \
    --publish  8890:8890 \
    -v "$(pwd)"/virtuoso-db:/database \
    -v /srv/data/abiram/kg/dbpedia2:/import \
    openlink/virtuoso-opensource-7:latest

docker run \
    --name vos2 \
    -it \
    --tty \
    --rm \
    --env DBA_PASSWORD=dba \
    --env DAV_PASSWORD=dba \
    --publish 1112:1111 \
    --publish  8891:8890 \
    -v /srv/data/abiram/triplestore/dbpediaVirt17-2:/database \
    -v /srv/data/abiram/LSQdata/dbpedia17-2:/import \
    openlink/virtuoso-opensource-7:latest
```
# Data loading
- docker exec -it vos isql 1111
//-  SPARQL create GRAPH <http://www.purl.com/Dbpedia>;

(change DefaultGraph in virtuoso.ini if different default graph.) (maybe change .ttl with .nt)
```
docker exec -it vos /bin/bash
echo "delete from DB.DBA.load_list;" > /settings/load.isql
cd /import
for i in `ls `; do
   if [[ ${i} == *.ttl ]] ;
   then
        echo "DB.DBA.TTLP_MT (file_to_string_output ('/import/"${i}"'), '', 'http://www.purl.com/Dbpedia');" >> /settings/load.isql
   fi
done;


#echo "rdf_loader_run ();" >> /settings/load.isql
echo "checkpoint;" >> /settings/load.isql

/opt/virtuoso-opensource/bin/isql exec="LOAD /settings/load.isql" && exit
```

## Checks to see that the data has been loaded:
```
docker exec -it vos isql 1111
SPARQL SELECT * FROM <http://www.purl.com/Dbpedia> WHERE {?s ?p ?o};
SPARQL SELECT COUNT(*) FROM <http://www.purl.com/Dbpedia> WHERE {?s ?p ?q};
```