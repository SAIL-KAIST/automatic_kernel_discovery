import os

from traitlets.traitlets import default
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import click

from re import search
import mlflow

def run_an_entry(entrypoint, parameters=None):
    
    print(f"Running entrypoint: {entrypoint} with parameters: {parameters}")
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)
@click.command()
@click.option("--time-series")
@click.option("--start-time")
@click.option("--end-time")
def workflow(time_series, start_time, end_time):
    
    with mlflow.start_run(run_name="Main") as active_run:
        
        # 1. LOAD
        load_run = run_an_entry("load_data", )
        
        # 2. SEARCH
        data_file = os.path.join(load_run.info.artifact_uri, "data.pkl")
        search_run = run_an_entry("search_kernel", {"data_file": data_file})
        
        # 3. ANALYSIS
        model_file = os.path.join(search_run.info.artifact_uri, "model.pkl")
        analysis_run = run_an_entry("analysis", {"model_file": model_file})
        

if __name__ == '__main__':
    
    workflow()