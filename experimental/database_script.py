import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
import django

django.setup()

from web.models_abcd import *

def add_experiment(time_series, run_id):
    
    query_set = TimeSeries.objects.filter(short_name=time_series)
    
    if len(query_set) == 0:
        # if no such time series in DB, save the time series
        raise ValueError("No such time series in DB")
    else:
        time_series = query_set[0]
    
    report = Report(time_series=time_series, 
                    title="New experiment",
                    datetime="2021-07-21",
                    run_id=run_id)
    
    report.save()
    
    print(f"Successfully save a new report for time series [{time_series}] with run id [{run_id}]")
    

if __name__ == '__main__':
    
    time_series = "MSFT"
    run_id = "32269de1d70b4fbda9e6db2efae5c6d9"
    add_experiment(time_series, run_id)
    
    
    