import os
import sys
import click
import mlflow
from util import export

from mlflow.tracking.fluent import _get_experiment_id
from mlflow.entities import RunStatus
from mlflow.utils import mlflow_tags
from mlflow.utils.logging_utils import eprint

sys.path.append("..")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_if_already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    
    client = mlflow.tracking.MlflowClient()
    
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        
        # filter entry point
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        
        # check if parameters are the same
        parameter_matched = True
        for key, value in parameters.items():
            run_value = full_run.data.params.get(key)
            if run_value != value:
                parameter_matched = False
                break
        
        if not parameter_matched:
            continue
        
        # check if run is finished
        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(f"Found run but not FINISHED: (run_id={run_info.run_id}, status={run_info.status})")
            continue
        
        if git_commit != tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None):
            eprint(f"Found run but has a different source version")
            continue
        
        return client.get_run(run_info.run_id)
    
    eprint("No run found!")
    return None


def run_an_entry(entrypoint, parameters, git_commit, use_cache=True):
    
    existing_run = check_if_already_ran(entrypoint, parameters, git_commit)
    
    if use_cache and existing_run:
        eprint(f"Found existing run for entrypoint: {entrypoint} with parameters: {parameters}")
        return existing_run
    
    eprint(f"Lauching new run for entrypoint: {entrypoint} with parameters: {parameters}")
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option("--time-series")
@click.option("--start-time")
@click.option("--end-time")
@click.option("--use-cache", default=True, type=bool)
def workflow(time_series, start_time, end_time, use_cache):
    
    with mlflow.start_run(run_name="Main") as active_run:
        
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        
        # 1. LOAD
        load_run = run_an_entry("load_data", {"time_series": time_series, "start_time":start_time, "end_time":end_time}, git_commit, use_cache)
        mlflow.set_tag("load_run", load_run.info.run_id)
        
        # 2. SEARCH
        data_file = os.path.join(load_run.info.artifact_uri, "data.pkl")
        search_run = run_an_entry("search_kernel", {"data_file": data_file}, git_commit, use_cache)
        mlflow.set_tag("search_run", search_run.info.run_id)
        
        # 3. ANALYSIS
        model_file = os.path.join(search_run.info.artifact_uri, "model.pkl")
        analysis_run = run_an_entry("analysis", {"model_file": model_file}, git_commit, use_cache)
        mlflow.set_tag("analysis_run", analysis_run.info.run_id)
        
        # FINALIZE
        export(run_id=active_run.info.run_id, 
               search_id=search_run.info.run_id,
               analysis_id=analysis_run.info.run_id)
        
        
        
        
if __name__ == '__main__':
    
    workflow()