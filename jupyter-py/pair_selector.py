import statsmodels.tsa.stattools as smts
import statsmodels.api as sm

import itertools

def coint(df, intercept = True, sig_level = 0.01):
    """
    Find pairs (of 2 time series) that passes the cointegration test.
    ----------
    Parameters
    ----------
    df: a dataframe, each column is the time series of a certain stock
    
    intercept: 
        if True, OLS and ADF test are done manually
        if False, the coint() function from statsmodels.tsa.stattools, which does not include 
        intercept term while doing OLS regression, is used.
    
    sig_level: if p_value of cointegration test is below this level, then we can reject
    the NULL hypothesis, which says that the two series are not cointegrated
    
    ----------
    Return
    ----------
    A list of tuples of the form (name of stock 1, name of stock 2, p_value of cointegration test).
    
    """
    cointegrated_pairs = []
    
    stock_names = df.columns.values.tolist()
    N = len(stock_names)
    
    stock_pairs = list(itertools.combinations(stock_names, 2))
    
    for pair in stock_pairs:
        stock_1, stock_2 = pair

        p_value = 0

        if not intercept:
            p_value = smts.coint(df[stock_1].values, df[stock_2].values, trend='c')[1]
        else:
            Y = df[stock_1]
            X = df[stock_2]
            X = sm.add_constant(X)

            model = sm.OLS(Y, X)
            results = model.fit()

            p_value = smts.adfuller(results.resid)[1]

        if p_value < sig_level:
            cointegrated_pairs.append(tuple([stock_1, stock_2, p_value]))

    return cointegrated_pairs

def distance(df, threshold = 10):
    
    
    
















