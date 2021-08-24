from time import monotonic
import click
import mlflow
from util import load_pickle, save

import sys
sys.path.append("..")

from kernel_discovery.description.describe import ProductDesc

@click.command()
@click.option("--model-file")
@click.option("--component-file")
def generate_report(model_file, component_file):
    with mlflow.start_run(run_name="Generate Report") as run:
        model = load_pickle(model_file)
        x, y, ast, noise = model
        components = load_pickle(component_file)
        
        for component in components:
            kernel = component["kernel"]
            monotonic = component["monotonic"]
            gradient = component["gradient"]
            description = ProductDesc(prod=kernel, x=x, monotonic=monotonic, gradient=gradient)
            summary, full_desc, extrap_desc = description.translate()
            
            component["summary"] = summary
            component["full_desc"] = full_desc
            component["extrap_desc"] =extrap_desc
        
        mlflow.log_artifact(save(components, "components.pkl"))
        
    
if __name__ == '__main__':
    generate_report()