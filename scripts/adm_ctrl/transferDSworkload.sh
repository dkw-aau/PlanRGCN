NN_workload="/data/DBpedia_3_class_full/load_balance10ClientsQPP2/load_balance_NN1/workload.pck"
SVM_workload="/data/DBpedia_3_class_full/load_balance10ClientsQPP2/load_balance_SVM1/workload.pck"
plan_workload="/data/DBpedia_3_class_full/load_balance10ClientsQPP2/load_balance_Plan1/workload.pck"

outFolder="/data/DBpedia_3_class_full/admission_control"
WORKLOAD=6
BASEOUTPUT=$outFolder/workload$WORKLOAD
#mkdir -p $BASEOUTPUT
METHOD="nn"
mkdir -p "$BASEOUTPUT/${METHOD}"
cp $NN_workload "$BASEOUTPUT/${METHOD}"
METHOD="planrgcn"
mkdir -p "$BASEOUTPUT/${METHOD}"
cp $plan_workload "$BASEOUTPUT/${METHOD}"
METHOD="svm"
mkdir -p "$BASEOUTPUT/${METHOD}"
cp $SVM_workload "$BASEOUTPUT/${METHOD}"