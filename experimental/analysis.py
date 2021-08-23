import os
import pickle
import click
import mlflow
import tempfile
from util import save, load_pickle

# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.analysis import Result

@click.command()
@click.option("--model-file")
def analysis(model_file):
    with mlflow.start_run(run_name="Analysis") as run:
        model = load_pickle(model_file)
        x, y, ast, noise = model
        result = Result(x, y, ast, noise)
        components = result.order_by_mae_reduction()
        
        save_file = save(components, name="components.pkl")
        mlflow.log_artifact(save_file)
        
        # generate plots and log them
        raw, fit, sample = result.full_posterior_plot()
        mlflow.log_figure(raw, "raw.png")
        mlflow.log_figure(fit, "fit.png")
        mlflow.log_figure(sample, "sample.png")

if __name__ == '__main__':
    
    analysis()
    