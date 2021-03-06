from kernel_discovery.description.simplify import distribution
import os
import pickle
import click
import mlflow
from util import save, load_pickle

# append working directory to Python path
import sys
sys.path.append("..")
from kernel_discovery.analysis import Result, IndividualAnalysis, CummulativeAnalysis, ModelCheckingAnalysis
from kernel_discovery.description.describe import ProductDesc, model_checking_report

import matplotlib.pyplot as plt


def individual_analysis(x, y, ast, noise, components):
    
    result = IndividualAnalysis(x, y, ast, noise)
    result.load_components(components)
    
    list_figs = result.analyze()
        
    for i, figs in enumerate(list_figs):
        fit_fig, extrap_fig, sample_fig = figs
        mlflow.log_figure(fit_fig, f"fit_{i}.png")
        mlflow.log_figure(extrap_fig, f"extrap_{i}.png")
        mlflow.log_figure(sample_fig, f"sample_{i}.png")
        
        for fig in figs:
            plt.close(fig)
    
    return result.components

def cummulative_analysis(x, y, ast, noise, components):
    
    result = CummulativeAnalysis(x, y, ast, noise)
    result.load_components(components)
    
    list_figs = result.analyze()
    
    for i, figs in enumerate(list_figs):
        cumm_fit_fig, cumm_extrap_fig, cumm_sample_fig, anti_res_fig = figs
        mlflow.log_figure(cumm_fit_fig, f"cum_fit_{i}.png")
        mlflow.log_figure(cumm_extrap_fig, f"cum_extrap_{i}.png")
        mlflow.log_figure(cumm_sample_fig, f"cum_sample_{i}.png")
        if anti_res_fig:
            mlflow.log_figure(anti_res_fig, f"anti_res_{i}.png")
        
        for fig in figs:
            plt.close(fig)
    
    return result.components
    
def model_check_analysis(x, y, ast, noise, components):

    result = ModelCheckingAnalysis(x, y, ast, noise)
    result.load_components(components)
    list_figs = result.analyze()
    
    for i, figs in enumerate(list_figs):
        qq_fig, acf_fig, pxx_fig = figs
        # mlflow.log_figure(mmd_fig, f"mmd_{i}.png")
        mlflow.log_figure(qq_fig, f"qq_band_{i}.png")
        mlflow.log_figure(acf_fig, f"acf_band_{i}.png")
        mlflow.log_figure(pxx_fig, f"pxx_band_{i}.png")
        
        for fig in figs:
            plt.close(fig)
        
    return result.components

def make_report(x, components):

    
    for component in components:
        kernel = component["kernel"]
        monotonic = component["monotonic"]
        gradient = component["gradient"]
        description = ProductDesc(prod=kernel, x=x, monotonic=monotonic, gradient=gradient)
        summary, full_desc, extrap_desc = description.translate()
        
        component["summary"] = summary
        component["full_desc"] = ".\n".join(full_desc)
        component["extrap_desc"] =".\n".join(extrap_desc)
        
        # model checking report
        discussion, bad_fit = model_checking_report(component)
        component["discussion"] = discussion
        component.update(bad_fit)
    
    return components
    

@click.command()
@click.option("--model-file")
def analysis(model_file):
    with mlflow.start_run(run_name="Analysis") as run:
        model = load_pickle(model_file)
        x, y, ast, noise = model
        result = Result(x, y, ast, noise)
        components = result.order_by_mae_reduction()
        
        
        # generate plots and log them
        raw, fit, sample = result.full_posterior_plot()
        mlflow.log_figure(raw, "raw.png")
        mlflow.log_figure(fit, "fit.png")
        mlflow.log_figure(sample, "sample.png")
        
        # individual
        components = individual_analysis(x, y, ast, noise, components)
        # cummulative
        components = cummulative_analysis(x, y, ast, noise, components)
        # model checking # TURN OFF FOR NOW
        components = model_check_analysis(x, y, ast, noise, components)
        #generate report
        components = make_report(x, components)
        
        # remove kernel object in each component to prevent pickle loading in the website
        for component in components:
            del component["kernel"] 
        
        save_file = save(components, name="components.pkl")
        mlflow.log_artifact(save_file)
        

if __name__ == '__main__':
    # # debug
    # model_file = "/home/anhth/projects/automatic_news/experimental/mlruns/0/2401bc4e71524765b1a05b6f32d51dfe/artifacts/model.pkl"
    # analysis(model_file)
    
    # main
    analysis()
    