import os
import click
import tempfile
import pickle
import mlflow

# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.analysis import IndividualAnalysis

@click.command()
@click.option("--model-file")
@click.option("--component-file")
def individual_analysis(model_file, component_file):
    with mlflow.start_run() as run:
        model = load_pickle(model_file)
        x, y, ast, noise = model
        result = IndividualAnalysis(x, y, ast, noise)
        components = load_pickle(component_file)
        result.load_components(components)
        
        components, list_figs = result.analyze()
        
        mlflow.log_artifact(save(components))
        
        for i, figs in enumerate(list_figs):
            fit_fig, extrap_fig, sample_fig = figs
            mlflow.log_figure(fit_fig, f"fit_{i}.png")
            mlflow.log_figure(extrap_fig, f"extrap_{i}.png")
            mlflow.log_figure(sample_fig, f"sample_{i}.png")
            

def load_pickle(file):
    
    with open(file, "rb") as f:
        result = pickle.load(f)
    
    return result

def save(components):
    
    tempdir = tempfile.mkdtemp()
    save_file = os.path.join(tempdir, "components.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(components, f)
    
    print(f"Save components to temporary file {save_file}")
    
    return save_file
    
    

if __name__ == '__main__':
    
    individual_analysis()