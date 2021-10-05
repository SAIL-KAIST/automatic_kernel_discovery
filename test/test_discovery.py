from kernel_discovery.discovery import ABCDiscovery
from kernel_discovery.io import retrieve

def test_abcd_discovery():
    
    ticker = 'MSFT'
    start = '2021-01-01'
    end = '2021-07-01'
    x, y, ticker = retrieve(ticker_name=ticker, start=start, end=end)

    discovery = ABCDiscovery(cluster_kwargs=dict(cluster=None))

    results = discovery.discover(x, y)
    print(results)