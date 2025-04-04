
queries=$1
#mvn package -f '/PlanRGCN/qpe/pom.xml' > /dev/null
#mvn package -f '/PlanRGCN/qpe/pom.xml'
jarfile=/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar

train=$queries/train_sampled.tsv
val=$queries/val_sampled.tsv
test=$queries/test_sampled.tsv
#outputPath=/PlanRGCN/extracted_features/queryplans
outputPath=$queries/queryplans #/qpp/dataset/DBpedia2016limitless/queryplans
mkdir -p $outputPath
task=extract-query-plans

java -jar $jarfile $task $train $outputPath "lsq=true" &>> query_plan_gen_log.txt
java -jar $jarfile $task $val $outputPath "lsq=true" &>> query_plan_gen_log.txt
java -jar $jarfile $task $test $outputPath "lsq=true" &>> query_plan_gen_log.txt
#mv query_plan_gen_log.txt $outputPath/..
#(mvn exec:java -f "/PlanRGCN/qpe/pom.xml" -Dexec.args="$task $queries $outputPath")

#bash run.sh > query_plan_gen_log.txt 2>&1
echo "Script finnished after $SECONDS s"