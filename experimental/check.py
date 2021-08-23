import click
import mlflow
from util import save, load_pickle

# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.analysis import ModelCheckingAnalysis

@click.command()
@click.option("--model-file")
@click.option("--component-file")
def model_checking(model_file, component_file):
    model = load_pickle(model_file)
    x, y, ast, noise = model
    result = ModelCheckingAnalysis(x, y, ast, noise)
    components = load_pickle(component_file)
    result.load_components(components)
    list_figs = result.analyze()
    
    for i, figs in enumerate(list_figs):
        mmd_fig, qq_fig, acf_fig, pxx_fig = figs
        mlflow.log_figure(mmd_fig, f"mmd_{i}.png")
        mlflow.log_figure(qq_fig, f"qq_band_{i}.png")
        mlflow.log_figure(acf_fig, f"acf_band_{i}.png")
        mlflow.log_figure(pxx_fig, f"pxx_band_{i}.png")
        
    mlflow.log_artifact(save(result.components, "components.pkl"))
        

if __name__ == '__main__':
    
    model_checking()