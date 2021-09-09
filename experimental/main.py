import os
import sys
import click
import mlflow
from util import export

sys.path.append("..")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        
        mlflow.log_param("time series", time_series)
        mlflow.log_param("start time", start_time)
        mlflow.log_param("end time", end_time)
        
        # 1. LOAD
        load_run = run_an_entry("load_data", {"time_series": time_series, "start_time":start_time, "end_time":end_time})
        mlflow.set_tag("load_run", load_run.info.run_id)
        
        # 2. SEARCH
        data_file = os.path.join(load_run.info.artifact_uri, "data.pkl")
        search_run = run_an_entry("search_kernel", {"data_file": data_file})
        mlflow.set_tag("search_run", search_run.info.run_id)
        
        # 3. ANALYSIS
        model_file = os.path.join(search_run.info.artifact_uri, "model.pkl")
        analysis_run = run_an_entry("analysis", {"model_file": model_file})
        mlflow.set_tag("analysis_run", analysis_run.info.run_id)
        
        # FINALIZE
        export(run_id=active_run.info.run_id, 
               search_id=search_run.info.run_id,
               analysis_id=analysis_run.info.run_id)
        
        
        
        
if __name__ == '__main__':
    
    workflow()