from collections import deque
from load_balance.workload.query import Query
from load_balance.workload.worker import Worker

class Queue:
    def __init__(self):
        self.queries = deque()
        self.worker :list[Worker]= list()
        
    def add_worker(self, worker):
        self.worker.append(worker)
    
    def add(self, query:Query):
        self.queries.append(query)
        
    def pop(self):
        return self.queries.popleft()

    def step(self, time):
        for w in self.worker:
            w.step(time)
        for w in self.worker:
            if w.is_free() and len(self)> 0:
                val = w.assign_query(self.pop(), time)
                if not val:
                    raise Exception("Something wrong here")
    
    def __len__(self):
        return len(self.queries)
    def is_finished(self):
        is_worker_finished = True
        for w in self.worker:
            if not w.is_free():
                is_worker_finished=False
        if len(self) == 0 and is_worker_finished:
            return True
        return False
