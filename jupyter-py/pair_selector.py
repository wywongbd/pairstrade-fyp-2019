import statsmodels.tsa.stattools as smts
import statsmodels.api as sm
import numpy as np
import itertools
import matplotlib.pyplot as plt

def coint(df, intercept = True, sig_level = 0.01):
    """
    Find pairs (of 2 time series) that passes the cointegration test.

    Parameters
    ----------
    df: pandas dataframe
        each column is the time series of a certain stock
    
    intercept: boolean
        if True, OLS and ADF test are done manually
        if False, the coint() function from statsmodels.tsa.stattools, which 
        does not include intercept term while doing OLS regression, is used.
    
    sig_level: if p_value of cointegration test is below this level, then we
        can reject the NULL hypothesis, which says that the two series are not
        cointegrated
    
    Return
    ------
    A list of tuples of the form (name of stock 1, name of stock 2, p_value of
    cointegration test).
    
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

def select_pairs_for_all_combin(train_df, test_df, config, scoreF, plot=True):
    """
    Find the desired pairs of stock name either by thresholding the score of
    each pair or pick the first n pairs with the lowest score.

    Parameters
    ----------
    train_df: pandas dataframe
        For training data, each column is the time series of a stock
    test_df: pandas dataframe
        For testing data, each column is the time series of a stock
    config: map
        It should contains either "threshold" or "n". The value of "threshold"
        will be used to determine whether a particular pair will be returned.
        If the score of a pair is lower than the "threshold" value, the pair
        will be returned as desired pair. The value of "n" will indicate the
        maximum number of pairs to return. The n returned pairs have the lowest
        scores.

        It should also contain a function associated with the key
        "score_function". The function should accept two series (in type pandas
        Series or numpy arrays) and return a score. The lower the score is
        indicates the better the pair are.

        It should contain another function associated with the key 
        "series_transform". The function should accept two pairs of series (in
        type pandas Series or numpy arrays). The first pair consists of two
        training price series and the second pair consists of two corresponding
        testing price series. This function should return two pairs of
        transformed series corresponding to the parameter pairs. Spread should
        be formed by subtracting one series by another within the same pair.
    plot: boolean
        if True, plot the returned pairs for visualization. Do not plot
        otherwise

    Return
    ----------
    A list of tuples of the form (name of stock 1, name of stock 2) sorted by
    distance in assending order.
    """

    # config checking
    if ("threshold" in config) ^ ("n" in config):
        raise Exception(
            "Please include either the key 'threshold' or 'n' in config.")
    elif "score_function" not in config:
        raise Exception("Please include a key 'score_function' in config.")
    elif "series_transform" not in config:
        raise Exception("Please include a key 'series_transform' in config.")
    score_function = config['score_function']
    series_transform = config['series_transform']
    
    stock_names = train_df.columns.values.tolist()
    N = len(stock_names)

    scores = np.zeros(N)
    pairs = []
    
    stock_pairs = list(itertools.combinations(stock_names, 2))
    i = 0

    # compute scores for all possible pairs
    for pair in stock_pairs:
        price_series = (train_df[pair[0]], train_df[pair[1]])

        score = score_function(*price_series)
        scores[i] = score

        pairs.append(pair)
        i += 1

    # obtain the result pairs by either thresholding the score or choose the
    # first n pairs
    result_pairs = []
    if "threshold" in config:
        result_indices = np.where(scores < config['threshold'])
        for i in result_indices:
            result_pairs.append(pairs[i])
    else:
        n = config['n']
        if len(scores) > n:
            first_n_indices = np.argpartition(scores, n)[:n]
            for i in first_n_indices:
                result_pairs.append(pairs[i])
        else:
            print('n is larger than the number of combinations of pairs!!!')
            result_pairs = pairs


    # plot for eyeballing
    if plot == True:
        for pair in result_pairs:
            training_price_series = (train_df[pair[0]], train_df[pair[1]])
            testing_price_series = (test_df[pair[0]], test_df[pair[1]])

            trans_training_series, trans_testing_series = \
                series_transform(training_price_series, testing_price_series)

            plot_two_series(*price_series, *pair, title='Training Phrase Data')

            plot_two_series(*trans_training_series, *pair,
                title='Normalized Training Price Series')

            plot_two_series(*trans_testing_series, *pair,
                title='Normalized Testing Price Series')

    return result_pairs


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

def distance_score(P1, P2):
        mean1, std1 = compute_stat(P1)
        mean2, std2 = compute_stat(P2)

        p1 = (P1 - mean1) / std1
        p2 = (P2 - mean2) / std2

        # compute distance
        diff = p1 - p2
        return (diff * diff).sum()


            P1, P2 = train_df[pair[0]].values, train_df[pair[1]].values

            mean1, std1 = compute_stat(P1)
            mean2, std2 = compute_stat(P2)

            p1 = (train_df[pair[0]] - mean1) / std1
            p2 = (train_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair,
                title='Normalized Training Price Series')

            p1 = (test_df[pair[0]] - mean1) / std1
            p2 = (test_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair,
                title='Normalized Testing Price Series')

    return results

def distance_transform(training_pair, testing_pair):
    training_P1, training_P2 = training_pair

    # compute_stat should not change the series
    mean1, std1 = compute_stat(training_pair[0])
    mean2, std2 = compute_stat(training_pair[1])

    p1 = (training_pair[0] - mean1) / std1
    p2 = (training_pair[1] - mean2) / std2

    trans_training = (p1, p2)

    p1 = (testing_pair[0] - mean1) / std1
    p2 = (testing_pair[1] - mean2) / std2

    trans_testing = (p1, p2)
    return trans_training, trans_testing

def intersection(train_df, test_df, n = 10, plot=True):
    """
    Find the closest n pairs (of 2 time series) computed based on their
    normalized price.

    Parameters
    ----------
    train_df: pandas dataframe
        for training data, each column is the time series of a certain stock
    test_df: pandas dataframe
        for testing data, each column is the time series of a certain stock
    n: int
        the number maximum number of pairs to return
    plot: boolean
        if True, plot the result for visualization. Do not plot otherwise

    Return
    ----------
    A list of tuples of the form (name of stock 1, name of stock 2) sorted by
    distance in assending order.
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
            plot_two_series(train_df[pair[0]], train_df[pair[1]], *pair,
                title='Training Phrase Data')

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
            plot_two_series(p1, p2, *pair,
                title='Normalized Training Price Series')

            p1 = (test_df[pair[0]] - mean1) / std1
            p2 = (test_df[pair[1]] - mean2) / std2
            plot_two_series(p1, p2, *pair,
                title='Normalized Testing Price Series')

    return results













