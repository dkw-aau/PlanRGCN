class Timer:
    def __init__(self):
        self.current_time = 0
        self.increment = 0.001 #1ms
        self.increment = 0.000001 #ultra small for instant step
        
    def step(self):
        self.current_time += self.increment
        
    def time(self):
        return self.current_time
