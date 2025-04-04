
DATASET=$1
BASELINEDIR="$DATASET"/baseline
QUERYLOG="$DATASET"/all.tsv
OUTPUTDIR=$2
CPUS=30
mkdir -p $OUTPUTDIR
mkdir -p $BASELINEDIR
cp -r "$DATASET"/*tsv $BASELINEDIR
{ time python3 -m qpp_features.feature_generator \
                $QUERYLOG \
                $OUTPUTDIR \
                -t combinations \
                -cpus $CPUS;
} 2> "$DATASET"/comb_time_20_cpu.log
echo "Combinations " $SECONDS

{ time python3 -m qpp_features.feature_generator \
                $QUERYLOG \
                $OUTPUTDIR \
                -t dist_calc \
                -cpus $CPUS;
} 2> "$DATASET"/dist_time_20_cpu.log
echo "dist_calc " $SECONDS

{ time python3 /PlanRGCN/qpp/qpp_features/qpp_features/database.py \
                "$OUTPUTDIR"/distances/ \
                "$BASELINEDIR"/ged.db
} 2> "$DATASET"/ged_db_creation.log
echo "DBCreation " $SECONDS

python3 -m qpp_new.feature_combiner $BASELINEDIR "$BASELINEDIR"/ged.db

#algebra features:
{ time python3 -m qpp_features.feature_generator \
                $QUERYLOG $DATASET -t alg_feat 1> "$DATASET"/baseline/alg_feat_time.log ; } 2> "$DATASET"/baseline/alg_feat_time.errlog
rm "$DATASET"/baseline/*txt
