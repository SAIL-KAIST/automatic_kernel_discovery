import click
import mlflow
from util import load_pickle, save

import sys
sys.path.append("..")
from kernel_discovery.analysis import CummulativeAnalysis

@click.command()
@click.option("--model-file")
@click.option("--component-file")
def cummulative_analysis(model_file, component_file):
    with mlflow.start_run(run_name="Cummulative Analysis") as run:
        model = load_pickle(model_file)
        x, y, ast, noise = model
        result = CummulativeAnalysis(x, y, ast, noise)
        components = load_pickle(component_file)
        result.load_components(components)
        
        list_figs = result.analyze()
        
        for i, figs in enumerate(list_figs):
            cumm_fit_fig, cumm_extrap_fig, cumm_sample_fig, anti_res_fig = figs
            mlflow.log_figure(cumm_fit_fig, f"cum_fit_{i}.png")
            mlflow.log_figure(cumm_extrap_fig, f"cum_extrap_{i}.png")
            mlflow.log_figure(cumm_sample_fig, f"cum_sample_{i}.png")
            if anti_res_fig:
                mlflow.log_figure(anti_res_fig, f"anti_res_{i}.png")
                

        saved_components = save(result.components, "components.pkl")
        mlflow.log_artifact(saved_components)
    
    

if __name__ == '__main__':
    cummulative_analysis()
    