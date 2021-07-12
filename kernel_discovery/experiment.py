from kernel_discovery.discovery import ABCDiscovery
import logging
import uuid
from kernel_discovery.io import retrieve
from kernel_discovery.postprocessing import make_all_figures_and_report

class Experiment():
    
    def __init__(self, time_series, start_time, end_time) -> None:
        
        self.uuid = str(uuid.uuid1())
        self.time_series = time_series
        self.start_time, self.end_time =  start_time, end_time
        
        self.logger = logging.getLogger(type(self).__name__)
    
    def get_data(self):
        self.logger.info(f"Get data for time series: {self.time_series}")
        # if this time series is new save to DB
        self.x, self.y, self.ticker = retrieve(self.time_series, self.start_time, self.end_time)
    
    def search(self):
        self.logger.info("Start the kernel search ")
        searcher = ABCDiscovery()
        results = searcher.discover(self.x, self.y)
        # TODO: save results to some files
        results = list(results.values())
        self.best_kernel = results[0]["ast"]
        self.optimized_noise = results[0]["noise"]
    
    def post_process(self):
        self.logger.info("Perform post processing")
        make_all_figures_and_report(x=self.x,
                                    y=self.y,
                                    kernel=self.best_kernel,
                                    noise=self.optimized_noise,
                                    experiment=self)
    
    def run(self):
        
        self.get_data()
        
        self.search()
        
        self.post_process()
        
if __name__ == '__main__':
    
    exp = Experiment(time_series='MSFT', start_time='2021-01-01', end_time='2021-03-01')
    exp.run()
    