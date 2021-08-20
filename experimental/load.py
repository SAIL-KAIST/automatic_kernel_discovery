import os
import click
import pickle
import mlflow
import tempfile
# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.io import retrieve
from kernel_discovery.preprocessing import preprocessing


@click.command()
@click.option("--time-series", type=str)
@click.option("--start-time", type=str)
@click.option("--end-time", type=str)
def load(time_series, start_time, end_time):
    print(f"Get data for time series: {time_series}")
    with mlflow.start_run() as run:
        x, y, ticker = retrieve(time_series, start_time, end_time)
        
        x, y = preprocessing(x, y, rescale_x_to_upper_bound=None)
    
        local_dir = tempfile.mkdtemp()
        
        data_file = os.path.join(local_dir, "data.pkl")
        
        with open(data_file, "wb") as f:
            pickle.dump([x, y], f)
        
        print(f"Save data to {data_file}")
        mlflow.log_artifact(data_file)
    

if __name__ == '__main__':
    load()