import os
import click
import pickle
import mlflow
import tempfile
import tensorflow as tf
from gpflow.models.gpr import GPR

# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.discovery import ABCDiscovery
from kernel_discovery.description.transform import ast_to_text, ast_to_kernel

# config for search
SEARCH_DEPTH=6
CLUSTER_KWARGS = dict(cluster=None)


@click.command()
@click.option("--data-file")
@click.option("--search-depth", type=int)
def search(data_file, search_depth):
    
    with mlflow.start_run(run_name="Search") as run:
        x, y = load_data(data_file)
        
        searcher = ABCDiscovery(search_depth=SEARCH_DEPTH, 
                                cluster_kwargs=CLUSTER_KWARGS)
        results = searcher.discover(x, y)
        results = list(results.values())
        best_kernel = results[0]["ast"]
        optimized_noise = results[0]["noise"]
        best_score = results[0]["score"]
        
        model_dir = save_model(x, y, best_kernel, optimized_noise)
        mlflow.log_artifact(model_dir)
        mlflow.log_text(ast_to_text(best_kernel), "kernel.txt")
        mlflow.log_metric("noise", float(optimized_noise))
        mlflow.log_metric("score", best_score)
    
def save_model(x, y, ast, noise):
        
    local_dir = tempfile.mkdtemp()
    save_dir = os.path.join(local_dir, "model.pkl")
    with open(save_dir, "wb") as f:
        pickle.dump([x, y, ast, noise], f)
    print(f"Save model to a temporary dir {local_dir}")
    return save_dir

def load_data(data_file):
    
    with open(data_file, 'rb') as f:
        result = pickle.load(f)
        
    return result[0], result[1]
    
if __name__ == '__main__':
    
    search()