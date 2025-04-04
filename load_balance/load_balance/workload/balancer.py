from load_balance.workload.worker import Worker
from load_balance.workload.timer import Timer
from load_balance.workload.dat_struct import Queue
from load_balance.workload.workload import Workload

class Balancer:
    def __init__(self, workload):
        self.workload: Workload = workload
        self.timer = Timer()
        self.slow_queue = Queue()
        self.med_queue = Queue()
        self.fast_queue = Queue()
        
        #self.workers_fast:list[Worker] = []
        #self.workers_medium:list[Worker] = []
        #self.workers_slow:list[Worker] = []
        #self.worker_load = {}
        self.configure_worker_1()
    
    def configure_worker_1(self):
        for _ in range(3):
            worker =Worker()
            self.fast_queue.add_worker(worker)
            #self.worker_load[worker] = 0
        for _ in range(1):
            worker =Worker()
            self.med_queue.add_worker(worker)
            #self.worker_load[worker] = 0
        for _ in range(1):
            worker =Worker()
            self.slow_queue.add_worker(worker)
            #self.worker_load[worker] = 0
    
    def enque(self, q, a):
        q.arrivaltime = a
        match q.time_cls:
            case 0:
                self.fast_queue.add(q)
            case 1:
                self.med_queue.add(q)
            case 2:
                self.slow_queue.add(q)
            case _:
                raise Exception("Should be one on 0,1,2")
        
    
    def run(self):
        q, a = self.workload.pop()
        while (True):
            time = self.timer.time()
            while (a is not None and time < a):
                self.timer.step()
                time = self.timer.time()
            
            #when workload has been loaded into queues.
            if a is None:
                self.timer.step()
                time = self.timer.time()
            
            while a is not None and a < time:
                self.enque(q,a)
                q,a = self.workload.pop()
            self.fast_queue.step(time)
            self.med_queue.step(time)
            self.slow_queue.step(time)
            #self.timer.step()
            if (len(self.workload) == 0 and self.slow_queue.is_finished() and self.med_queue.is_finished() and self.fast_queue.is_finished()):
                break

class FIFOBalancer:
    def __init__(self, workload):
        self.workload: Workload = workload
        self.timer = Timer()
        self.fast_queue = Queue()
        
        self.configure_worker_1()
    
    def configure_worker_1(self):
        for _ in range(5):
            worker =Worker()
            self.fast_queue.add_worker(worker)
    
    def enque(self, q, a):
        q.arrivaltime = a
        self.fast_queue.add(q)
        
    
    def run(self):
        q, a = self.workload.pop()
        while (True):
            time = self.timer.time()
            while (a is not None and time < a):
                self.timer.step()
                time = self.timer.time()
            
            #when workload has been loaded into queues.
            if a is None:
                self.timer.step()
                time = self.timer.time()
            
            while a is not None and a < time:
                self.enque(q,a)
                q,a = self.workload.pop()
            self.fast_queue.step(time)
            
            if (len(self.workload) == 0 and self.fast_queue.is_finished()):
                break
