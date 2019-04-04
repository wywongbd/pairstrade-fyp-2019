import backtrader as bt
import backtrader.feeds as btfeeds

class PandasData(bt.feed.DataBase):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''

    params = (
        ('datetime', 0),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', 1),
        ('volume', -1),
        ('openinterest', -1),
    )