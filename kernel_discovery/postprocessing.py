from kernel_discovery.kernel import Periodic
from kernel_discovery.description.transform import kernel_to_ast
import numpy as np
from anytree import Node


from kernel_discovery.description.plot import Result
from kernel_discovery.description.describe import produce_summary


import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
import django
from django.conf import settings

django.setup()


from web.models_abcd import *


def make_all_figures_and_report(x: np.array, y: np.array, kernel: Node, noise: np.array, experiment):
    
    save_dir = os.path.join(settings.FIGURE_FOLDER, experiment.uuid)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        raise RuntimeError("figure folder is already existed. Duplicated Experiment UUID!")
    
    result = Result(x, y, kernel, noise, save_dir=save_dir)
    result.process()
    
    for component in result.components:
        component.make_description()
    
    
    # save generated report to DB
    query_set = TimeSeries.objects.filter(short_name=experiment.time_series)
    
    if len(query_set) == 0:
        # if no such time series in DB, save the time series
        short_name = experiment.ticker.info['symbol']
        name = experiment.ticker.info['shortName']
        time_series = TimeSeries(name=name, short_name=short_name)
        time_series.save()
    else:
        time_series = query_set[0]
    
    report = Report(time_series=time_series, 
                    title="some title",
                    datetime="2021-07-21",
                    image_folder=experiment.uuid)
    
    report.save()
    
    for i, comp in enumerate(result.components):
        # intialize with statics
        web_comp = Component(report_id=report,
                              i=i,
                              var=comp.var,
                              mae=comp.mae,
                              mae_reduction=comp.mae_reduction,
                              cum_var=comp.cum_var,
                              cum_res_var=comp.cum_res_var,
                              )
        
        # model checking
        web_comp.mmd_p_value = comp.mmd_p_value
        web_comp.acf_min = comp.acf_min
        web_comp.acf_min_loc = comp.acf_min_loc
        web_comp.pxx_max = comp.pxx_max
        web_comp.pxx_max_loc = comp.pxx_max_loc
        web_comp.qq_d_max = comp.qq_d_max
        web_comp.qq_d_min = comp.qq_d_min
        
        # add image info
        web_comp.fit = comp.fit
        web_comp.extrap = comp.extrap
        web_comp.sample = comp.sample
        web_comp.cum_fit = comp.cum_fit
        web_comp.cum_sample = comp.cum_sample
        web_comp.anti_res = comp.anti_res
        
        # add description fields
        web_comp.summary = comp.summary
        web_comp.full_desc = comp.full_desc
        web_comp.extrap_desc = comp.extrap_desc
        
        web_comp.save()

    
    

if __name__ == '__main__':


    from kernel_discovery.kernel import Linear, RBF
    import uuid
    import os
    
    
    dataset_name = "Some Name"
    k = Linear() * RBF() + Periodic()
    prod = kernel_to_ast(k)
    
    x = np.linspace(0,1,100)[:,None]
    y = np.sin(x)    
    make_all_figures_and_report(x, y, prod, noise=np.array(0.1), experiment_id=str(uuid.uuid1()))
