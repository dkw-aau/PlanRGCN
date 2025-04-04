
EXCLUDESTR="--exclude ray_save --exclude ged.db --exclude distances --exclude distance "
DEST="."
DEST="/data/"
rsync -aWP -u qpp:/srv/data/abiram/data_qpp/wikidata_3_class_full $DEST $EXCLUDESTR
rsync -aWP -u qpp:/srv/data/abiram/data_qpp/DBpedia_3_class_full $DEST $EXCLUDESTR
