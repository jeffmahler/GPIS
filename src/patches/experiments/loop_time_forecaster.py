import time

class LoopTimingForecaster:
    
    def __init__(self, n):
        self._n = n
        self._i = 0
        self._start = None
        self._temp_start = None
        self._end = None
        
        self._times = []
        
    def _check_end(self):
        if self._end is not None:
            msg = "Can't perform this operation because end has already been called"
            logging.error(msg)
            raise Exception(msg)
                
    def report_forecast(self):
        self._check_end()
        if len(self._times) == 0:
            logging.info("No data on timing yet. Can't perform forecast")
        else:
            remaining_steps = self._n - self._i
            mean_time = sum(self._times) / len(self._times)
            remaining_time = mean_time * remaining_steps
            
            msg = "Remaining time: {0}, steps: {1}. Time per step: {2}".format(timedelta(seconds=remaining_time),
                                                                               remaining_steps, timedelta(seconds=mean_time))
            logging.info(msg)
            
    def record_loop_start(self):
        cur_time = time.time()
        self._check_end()
        if self._start is None:
            self._start = cur_time
            logging.info("Starting loop timer")
        self._temp_start = cur_time
    
    def record_loop_end(self):
        cur_time = time.time()
        self._check_end()
        
        if self._temp_start is None:
            msg = "Can't end loop timer because it hasn't been started!"
            logging.error(msg)
            raise Exception(msg)
        
        self._times.append(cur_time - self._temp_start)
        self._temp_start = None
        self._i += 1
        
    def record_end(self):
        self._end = time.time()