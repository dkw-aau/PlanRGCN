dispather () {
   until [ -f $1 ]
   do 
	   sleep 7200
   done
   bash $1
}

experiment_funct () {
   (cd ... && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
   touch $1
   sleep 30
   (cd ... && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)

}
experiment_runner () {
   (cd load_balance_FIFO_10 && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
   touch $1
   sleep 60
   (cd load_balance_PlanRGCN_10 && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
   touch $1
   sleep 60
   (cd load_balance_FIFO_22 && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
   touch $1
   sleep 60
   (cd load_balance_PlanRGCN_22 && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
   touch $1
   sleep 60
   (cd load_balance_FIFO_44 && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)0
   touch $1
   sleep 60
   (cd load_balance_PlanRGCN_44 && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
}
