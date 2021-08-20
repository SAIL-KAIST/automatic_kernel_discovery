import os
import pickle
import click
import mlflow
import tempfile
from gpflow.models.util import data_input_to_tensor

# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.analysis import Result

@click.command()
@click.option("--model-file")
def analysis(model_file):
    with mlflow.start_run() as run:
        x, y, ast, noise = load_model(model_file)
        
        result = Result(x, y, ast, noise)
        components = result.order_by_mae_reduction()
        
        save_file = save(components)
        mlflow.log_artifact(save_file)
        
        # generate plots and log them
        raw, fit, sample = result.full_posterior_plot()
        mlflow.log_figure(raw, "raw.png")
        mlflow.log_figure(fit, "fit.png")
        mlflow.log_figure(sample, "sample.png")
    

def load_model(model_dir):
    
    with open(model_dir, "rb") as f:
        result = pickle.load(f)
    
    x, y, ast, noise = result[0], result[1], result[2], result[3]
    return x, y, ast, noise

def save(components):
    
    tempdir = tempfile.mkdtemp()
    save_file = os.path.join(tempdir, "components.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(components, f)
    
    print(f"Save components to temporary file {save_file}")
    
    return save_file
    

if __name__ == '__main__':
    
    analysis()
    