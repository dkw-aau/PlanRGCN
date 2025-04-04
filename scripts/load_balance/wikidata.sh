experiment_runner_wiki () {
   LB=load_balanceSeed21 && basePath=/data/wikidata_0_1_10_v2_weight_loss
   cd $basePath
   (cd $LB/load_balanceSVM44 && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
   echo "SVM 44 10 worker"
   touch $1
   sleep 301
   (cd $LB/load_balanceNN44 && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
   echo "NN 44 10 worker"
   touch $1
   sleep 301
   (cd $LB/load_balancePlanRGCN44_10 && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
   echo "Plan 44 10 worker"
   touch $1
   sleep 301
   (cd $LB/load_balanceOracle44_10 && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
   echo "Oracle 44 10 worker"
   touch $1
   sleep 301
   (cd $LB/load_balanceFIFO44_10 && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
   echo "Oracle 44 10 worker"
   touch $1
}
experiment_runner_wiki restartVirt