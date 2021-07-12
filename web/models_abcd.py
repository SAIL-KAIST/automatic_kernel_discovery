
from django.db import models

class TimeSeries(models.Model):
    
    short_name = models.CharField(max_length=50)
    name = models.CharField(max_length=150)
    
    class Meta:
        db_table = "web_timeseries"
        managed = False
    

class Report(models.Model): 
    """
    Automatic generated reported from ABCD in English
    """
    
    time_series = models.ForeignKey(TimeSeries, on_delete=models.PROTECT)
    
    title = models.CharField(max_length=100)
    datetime = models.DateTimeField() # when to save to data base?
    image_folder = models.ImageField(upload_to="AutomaticNews/") # point to the folder that the generated html need
    
    class Meta:
        db_table = "web_report"
        managed = False
    
    def __str__(self) -> str:
        return self.title
    

class Component(models.Model):
    
    report_id = models.ForeignKey(Report, on_delete=models.CASCADE)
    
    # order of this component
    i = models.IntegerField()
    # statistic fields
    var = models.FloatField()
    
    mae = models.FloatField()
    mae_reduction = models.FloatField()
    
    # cumulative statistic fields
    cum_var = models.FloatField()
    cum_res_var = models.FloatField()
    
    
    # some more stastic for model checking
    # TODO: 
    
    # figure:
    # TODO: figure name seems to follows some patterns. So do not need to explicitly save it 
    fit= models.ImageField(upload_to="AutomaticNews", default='fit_0.png')
    extrap= models.ImageField(upload_to="AutomaticNews",default='extrap_0.png')
    sample= models.ImageField(upload_to="AutomaticNews",default='sample_0.png')
    cum_fit= models.ImageField(upload_to="AutomaticNews",default='cum_fit_0.png')
    cum_extrap= models.ImageField(upload_to="AutomaticNews",default='cum_extrap_0.png')
    cum_sample= models.ImageField(upload_to="AutomaticNews",default='cum_sample_0.png')
    anti_res= models.ImageField(upload_to="AutomaticNews",default='anti_res_0.png')
    
    
    # description fields
    summary = models.TextField()
    full_desc = models.TextField()
    extrap_desc = models.TextField()
    
    
    class Meta:
        db_table = "web_component"
    
    def __str__(self) -> str:
        return self.summary