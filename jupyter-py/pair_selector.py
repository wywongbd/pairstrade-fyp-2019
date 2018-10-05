import statsmodels.tsa.stattools as smts
import statsmodels.api as sm
import numpy as np
import itertools
import matplotlib.pyplot as plt

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

def compute_stat(p):
    return np.mean(p), np.std(p)

def plot_two_series(x1, x2, label1, label2, title, plt_width=10, plt_height=5):
    """
    Helper function for visualizing two series
    """
    plt.rcParams['figure.figsize'] = [plt_width, plt_height]
    plt.plot(x1, label=label1)
    plt.plot(x2, label=label2)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def distance(train_df, test_df, n = 10, plot=True):
    """
    Find the closest n pairs (of 2 time series) computed based on their normalized price.

    Parameters
    ----------
    train_df: a dataframe for training data, each column is the time series of a certain stock
    test_df: a dataframe for testing data, each column is the time series of a certain stock
    n: the number maximum number of pairs to return

    Return
    ----------
    A list of tuples of the form (name of stock 1, name of stock 2) sorted by distance in assending order.
    """
    
    scores_to_pairs = []
    
    stock_names = train_df.columns.values.tolist()
    N = len(stock_names)
    
    stock_pairs = list(itertools.combinations(stock_names, 2))
    
    for pair in stock_pairs:
        P1, P2 = train_df[pair[0]].values, train_df[pair[1]].values

        mean1, std1 = compute_stat(P1)
        mean2, std2 = compute_stat(P2)

        p1 = (P1 - mean1) / std1
        p2 = (P2 - mean2) / std2

        # compute distance
        diff = p1 - p2
        dist = (diff * diff).sum()
        scores_to_pairs.append((dist, pair))

    scores_to_pairs = sorted(scores_to_pairs, key=lambda x: x[0])

    results = None
    if len(scores_to_pairs) < n:
        results = [x[1] for x in scores_to_pairs]
    else:
        results = [x[1] for x in scores_to_pairs[:n]]

    # plot for eyeballing
    if plot == True:
        for pair in results:
            plot_two_series(train_df[pair[0]], train_df[pair[1]], *pair, title='Training Phrase Data')

            P1, P2 = train_df[pair[0]].values, train_df[pair[1]].values

            mean1, std1 = compute_stat(P1)
            mean2, std2 = compute_stat(P2)

            p1 = (train_df[pair[0]] - mean1) / std1
            p2 = (train_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair, title='Normalized Training Price Series')

            p1 = (test_df[pair[0]] - mean1) / std1
            p2 = (test_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair, title='Normalized Testing Price Series')

    return results


def intersection(train_df, test_df, n = 10, plot=True):
    """
    Find the closest n pairs (of 2 time series) computed based on their normalized price.

    Parameters
    ----------
    train_df: a dataframe for training data, each column is the time series of a certain stock
    test_df: a dataframe for testing data, each column is the time series of a certain stock
    n: the number maximum number of pairs to return

    Return
    ----------
    A list of tuples of the form (name of stock 1, name of stock 2) sorted by distance in assending order.
    """
    
    scores_to_pairs = []
    
    stock_names = train_df.columns.values.tolist()
    N = len(stock_names)
    
    stock_pairs = list(itertools.combinations(stock_names, 2))
    
    for pair in stock_pairs:
        P1, P2 = train_df[pair[0]].values, train_df[pair[1]].values

        # # build linear model
        # P1 = sm.add_constant(P1)
        # model = sm.OLS(P2, P1)
        # model_results = model.fit()

        # # get residul sign
        # residual_sign = np.sign(model_results.resid)

        # # compute number of intersection
        # rolled_sign = np.roll(residual_sign, 1)
        # rolled_sign[0] = 0
        # num_of_intersection = ((residual_sign * rolled_sign) == 1).sum()

        # scores_to_pairs.append((num_of_intersection, pair))

        mean1, std1 = compute_stat(P1)
        mean2, std2 = compute_stat(P2)

        p1 = (P1 - mean1) / std1
        p2 = (P2 - mean2) / std2

        # compute distance
        diff = p1 - p2

        # get residul sign
        residual_sign = np.sign(diff)

        # compute number of intersection
        rolled_sign = np.roll(residual_sign, 1)
        rolled_sign[0] = 0
        num_of_intersection = ((residual_sign * rolled_sign) == 1).sum()

        scores_to_pairs.append((num_of_intersection, pair))

    scores_to_pairs = sorted(scores_to_pairs, key=lambda x: x[0])

    results = None
    if len(scores_to_pairs) < n:
        results = [x[1] for x in scores_to_pairs]
    else:
        results = [x[1] for x in scores_to_pairs[:n]]

    # plot for eyeballing
    if plot == True:
        for pair in results:
            plot_two_series(train_df[pair[0]], train_df[pair[1]], *pair, title='Training Phrase Data')

            # P1, P2 = train_df[pair[0]].values, train_df[pair[1]].values

            # # build linear model
            # P1 = sm.add_constant(P1)
            # model = sm.OLS(P2, P1)
            # model_results = model.fit()

            # p1 = train_df[pair[0]] * model_results.params[1] + model_results.params[0]
            # p2 = train_df[pair[1]]
            # plot_two_series(p1, p2, *pair, title='Normalized Training Price Series')

            # p1 = test_df[pair[0]] * model_results.params[1] + model_results.params[0]
            # p2 = test_df[pair[1]]
            # plot_two_series(p1, p2, *pair, title='Normalized Testing Price Series')

            P1, P2 = train_df[pair[0]].values, train_df[pair[1]].values

            mean1, std1 = compute_stat(P1)
            mean2, std2 = compute_stat(P2)

            p1 = (train_df[pair[0]] - mean1) / std1
            p2 = (train_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair, title='Normalized Training Price Series')

            p1 = (test_df[pair[0]] - mean1) / std1
            p2 = (test_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair, title='Normalized Testing Price Series')

    print('updatedddddd')
    return results













