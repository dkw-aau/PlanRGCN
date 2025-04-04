import numpy as np


class ArrivalRateDecider:
    def __init__(self, distribution='poisson', seed = 42) -> None:
        self.distribution_type = distribution
        self.seed = seed
        np.random.seed(seed)
    
    def assign_arrival_rate(self, workload, mu=20):
        if self.distribution_type == 'poisson':
            return self.generative_exponential_arrivals(mu, len(workload))
            #return self.generate_arrivals(len(workload), mu=mu)
        pass
        
    def generate_arrivals(self, n,  samples, mu=2):
        #mu or lambda is the expected number of events occuring in a fixed-time interval.
        delays = np.random.poisson(mu,n)
        cu_delays = []
        total = 0
        for x in delays:
            cu_delays.extend([total]*x)
            total += 1
        return np.array(cu_delays)
    
    def generative_exponential_arrivals(self, number_of_queries_per_sec, samples):
        N = number_of_queries_per_sec #queries
        T=1 # symbolises per second
        lmbda= N/T
        count=samples
        y = np.random.exponential(1.0/lmbda, count)
        return np.cumsum(y)