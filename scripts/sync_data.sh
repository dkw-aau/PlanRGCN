#for qpp2 devserver container
EXCLUDESTR="--exclude ray_save --exclude distances --exclude combinations --exclude feature.log --exclude ged.db --exclude ged_db_creation.log"
RSYNCOPTIONES="avP" # old one "aWP"
if [ $# -eq 0 ]; then
  echo argument not supplied
elif [ ${1} == 'qpp2' ]; then
  rsync -$RSYNCOPTIONES katja20:/srv/data/tq74iz/data_qpp/DBpedia_3_class_full /data $EXCLUDESTR
  rsync -$RSYNCOPTIONES katja20:/srv/data/tq74iz/data_qpp/wikidata_3_class_full /data $EXCLUDESTR

  rsync -$RSYNCOPTIONES qpp5:/data/tq74iz/tq74iz/data_qpp/wikidata_3_class_full /data $EXCLUDESTR
  rsync -$RSYNCOPTIONES qpp5:/data/tq74iz/tq74iz/data_qpp/DBpedia_3_class_full /data $EXCLUDESTR

  rsync -$RSYNCOPTIONES qpp:/srv/data/abiram/data_qpp/DBpedia_3_class_full /data $EXCLUDESTR
  rsync -$RSYNCOPTIONES qpp:/srv/data/abiram/data_qpp/wikidata_3_class_full /data $EXCLUDESTR

elif [ ${1} == 'qpp1' ]; then
  rsync -$RSYNCOPTIONES qpp2:/data/abiram/data_qpp/DBpedia_3_class_full /data $EXCLUDESTR
  rsync -$RSYNCOPTIONES qpp2:/data/abiram/data_qpp/wikidata_3_class_full /data $EXCLUDESTR
else
  echo $1 is not valid
fi
