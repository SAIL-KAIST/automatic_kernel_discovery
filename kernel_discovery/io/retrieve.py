""" 
Use the library ```yfinance``` for Yahoo Finance API 
"""
import logging
import os
from pandas import to_datetime

import yfinance as yf

logger = logging.getLogger(os.path.basename(__file__))


def convert_date(date):
    time_tuple = date.timetuple()
    return float(time_tuple.tm_year) + float(time_tuple.tm_yday) / 365

def retrieve(ticker_name, start, end, select='Adj Close'):
    """
    Get data of a stock and return a numpy pairs
    """
    logger.info(f"Retrieve stock: {ticker_name} from [{start}] to [{end}] and select `{select}` column")
        
    data = yf.download(ticker_name, start=start, end=end)
    data.reset_index(level=0, inplace=True)    
    
    x = data['Date'].apply(lambda x: convert_date(to_datetime(x))).to_numpy()
    y = data[select].to_numpy()
    
    return x, y

if __name__ == "__main__":
        
    
    ticker = 'MSFT'
    start='2021-01-01'
    end = '2021-02-01'
    
    retrieve(ticker, start, end)