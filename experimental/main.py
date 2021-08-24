import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from re import search
import mlflow

def run_an_entry(entrypoint, parameters=None):
    
    print(f"Running entrypoint: {entrypoint} with parameters: {parameters}")
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

def workflow():
    
    with mlflow.start_run(run_name="Main") as active_run:
        
        # 1. LOAD
        load_run = run_an_entry("load_data")
        
        # 2. SEARCH
        data_file = os.path.join(load_run.info.artifact_uri, "data.pkl")
        search_run = run_an_entry("search_kernel", {"data_file": data_file})
        
        # 3. ANALYSIS
        model_file = os.path.join(search_run.info.artifact_uri, "model.pkl")
        analysis_run = run_an_entry("analysis", {"model_file": model_file})
        
        # 3.1 individual analysis
        component_file = os.path.join(analysis_run.info.artifact_uri, "components.pkl")
        individual_run = run_an_entry("analysis_individual", {"model_file": model_file, "component_file":component_file})
        
        # 3.2 cummulative analysis
        component_file = os.path.join(individual_run.info.artifact_uri, "components.pkl")
        cummulative_run = run_an_entry("analysis_cummulative", {"model_file": model_file, "component_file":component_file })
        
        # 3.3 Model checking
        component_file = os.path.join(cummulative_run.info.artifact_uri, "components.pkl")
        checking_run = run_an_entry("analysis_checking", {"model_file": model_file, "component_file":component_file})
        
        # 3.4 Generate report
        component_file = os.path.join(checking_run.info.artifact_uri, "components.pkl")
        run_an_entry("analysis_finalize", {"model_file":model_file, "component_file":component_file})
    

if __name__ == '__main__':
    
    workflow()