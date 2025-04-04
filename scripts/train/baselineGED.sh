

BASELINEDIR=/data/DBpediaV2
python3 -m qpp_new.feature_combiner \
                $BASELINEDIR /data/DBpediaV2/baseline/ged.db
echo "Finished after (DBPEDIA)"  $SECONDS
BASELINEDIR=/data/wikidataV2
python3 -m qpp_new.feature_combiner \
                $BASELINEDIR /data/wikidataV2/baseline/ged.db
echo "Finished after (wikidata)"  $SECONDS