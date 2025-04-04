lb_wikidata_qpp2 (){
	    config_path=/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
	        CPUS="10"
		    dbpath=/data/abiram/wdbench/virtuoso_dabase
		        imp_path=/data/abiram/wdbench/import
			    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
			        docker run -m 64g --rm --name wdbench_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v imp_path:/import -v $config_path:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:latest
}


lb_wikidata_qpp1 (){
    config_path=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
    dbpath=/srv/data/abiram/wdbench/virtuoso_dabase
    CPUS=8
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    docker run -m 64g --rm --name wdbench_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v /srv/data/abiram/wdbench/datafile:/import -v $config_path:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:latest
}

#ARG1 filepath  when the database is reaady to be restarted
#ARG2 filepath to when the 
# Needs ctr-c iterrupts to terminate
db_restarter (){
   while : 
   do
    until [ -f $1 ]
    do
        sleep 300 # checks every 5 min
    done
    #docker stop wdbench_virt
    docker stop dbpedia_virt
    rm $1
    #lb_wikidata_qpp1
    lb_dbpedia_qpp1
    echo Restarted
    sleep 5
    touch $2
   done
}

lb_dbpedia_qpp1 (){
    config_path=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_dbpedia_load_balance.ini
    dbpath=/srv/data/abiram/dbpediaKG/virtuoso-db-new2/virtuoso-db-new
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    docker run --name dbpedia_virt -it --tty --env DBA_PASSWORD=dba -m 64G --env DAV_PASSWORD=dba --publish 1113:1111 --publish 8892:8890 -v $dbpath:/database -v /srv/data/abiram/dbpedia2016/import_final:/import -v $config_path:/database/virtuoso.ini --cpus="10" openlink/virtuoso-opensource-7:latest
}
