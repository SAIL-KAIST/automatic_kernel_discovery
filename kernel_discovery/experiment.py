from kernel_discovery.description.transform import ast_to_text
import mlflow
from kernel_discovery.discovery import ABCDiscovery
import logging
import uuid
from kernel_discovery.io import retrieve
from kernel_discovery.postprocessing import make_all_figures_and_report
from kernel_discovery.preprocessing import preprocessing

class Experiment():
    
    def __init__(self, time_series, start_time, end_time, uri="") -> None:
        
        self.uuid = str(uuid.uuid1())
        self.time_series = time_series
        self.start_time, self.end_time =  start_time, end_time
        self.uri=uri
        self.logger = logging.getLogger(type(self).__name__)
        self.init_mlflow()
        
    def init_mlflow(self):
        # mlflow.create_experiment(name=self.time_series)
        # mlflow.set_tracking_uri(uri=self.uri)
        pass
        
    
    def get_data(self, preprocess=True):
        self.logger.info(f"Get data for time series: {self.time_series}")
        # if this time series is new save to DB
        self.x, self.y, self.ticker = retrieve(self.time_series, self.start_time, self.end_time)
        
        if preprocess:
            self.x, self.y = preprocessing(self.x, self.y, rescale_x_to_upper_bound=None)
    
    def search(self):
        self.logger.info("Start the kernel search ")
        searcher = ABCDiscovery(search_depth=3)
        results = searcher.discover(self.x, self.y)
        # TODO: save results to some files
        results = list(results.values())
        self.best_kernel = results[0]["ast"]
        self.optimized_noise = results[0]["noise"]
        self.best_score = results[0]["score"]
        mlflow.log_text(ast_to_text(self.best_kernel), "kernel.txt")
        mlflow.log_metric("noise", float(self.optimized_noise))
        mlflow.log_metric("score", self.best_score)
        self.logger.info(f"Search algorithm found \n \t {self.best_kernel} \n \t with score: {self.best_score:.3f}")
        
    
    def post_process(self):
        self.logger.info("Perform post processing")
        make_all_figures_and_report(x=self.x,
                                    y=self.y,
                                    kernel=self.best_kernel,
                                    noise=self.optimized_noise,
                                    experiment=self)
    
    def run(self):
        
        with mlflow.start_run() as run:
            
            mlflow.log_params(
                {"start_time": self.start_time,
                 "end_time": self.end_time,
                 } # TODO: log search param i.e, depth... 
            )
            
            self.get_data()
            
            self.search()
            
            self.post_process()
        
if __name__ == '__main__':
    
    exp = Experiment(time_series='MSFT', start_time='2021-01-01', end_time='2021-03-01')
    exp.run()
    