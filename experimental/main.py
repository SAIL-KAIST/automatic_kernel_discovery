import os
from re import search
import mlflow

def run_an_entry(entrypoint, parameters=None):
    
    print(f"Running entrypoint: {entrypoint} with parameters: {parameters}")
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

def workflow():
    
    with mlflow.start_run() as active_run:
        
        # 1. LOAD
        load_run = run_an_entry("load_data")
        
        # 2. SEARCH
        data_file = os.path.join(load_run.info.artifact_uri, "data.pkl")
        search_run = run_an_entry("search_kernel", {"data_file": data_file})
        
        # 3. ANALYSIS
        model_file = os.path.join(search_run.info.artifact_uri, "model.pkl")
        run_an_entry("analysis", {"model_file": model_file})
        
        # 3.1 
        run_an_entry("analysis_individual", {})
        
        # 3.2
        run_an_entry("analysis_cummulative", {})
        
        # 3.3
        run_an_entry("analysis_checking", {})
        
        # 3.4
        run_an_entry("analysis_finalize", {})
    

if __name__ == '__main__':
    
    workflow()