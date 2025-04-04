#WikidatA Workloads
workloadfilename=workload.pck
OUTPUTFOLDER="/data/wikidata_3_class_full/admission_controlV2/workload1"
mkdir -p "${OUTPUTFOLDER}/planrgcn"
mkdir -p "${OUTPUTFOLDER}/nn"
mkdir -p "${OUTPUTFOLDER}/svm"
cp /data/wikidata_3_class_full/admission_control/workload3_21/planrgcn/${workloadfilename} "${OUTPUTFOLDER}/planrgcn"
cp /data/wikidata_3_class_full/admission_control/workload2_21/nn/${workloadfilename} "${OUTPUTFOLDER}/nn"
cp /data/wikidata_3_class_full/admission_control/workload2_21/svm/${workloadfilename} "${OUTPUTFOLDER}/svm"

#DBpedia workloads
OUTPUTFOLDER="/data/DBpedia_3_class_full/admission_controlV2/workload1"
BASEPATH="/data/DBpedia_3_class_full/admission_control/workload6"
mkdir -p "${OUTPUTFOLDER}/planrgcn"
mkdir -p "${OUTPUTFOLDER}/nn"
mkdir -p "${OUTPUTFOLDER}/svm"
cp "${BASEPATH}/planrgcn/${workloadfilename}" "${OUTPUTFOLDER}/planrgcn"
cp "${BASEPATH}/nn/${workloadfilename}" "${OUTPUTFOLDER}/nn"
cp "${BASEPATH}/svm/${workloadfilename}" "${OUTPUTFOLDER}/svm"