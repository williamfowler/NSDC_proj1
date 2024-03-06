'''
Test Cases
--------
>>> y_N = 0.0
>>> yhat_N = 4.123
>>> calc_root_mean_squared_error(y_N, yhat_N)
4.123

>>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
>>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
>>> rmse = calc_root_mean_squared_error(y_N, yhat_N)
>>> np.round(rmse, 6)
1.154701
'''

import numpy as np


def calc_root_mean_squared_error(y_N, yhat_N):
    ''' Compute root mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    rmse : scalar float
        Root mean squared error performance metric
        .. math:
            rmse(y,\hat{y}) = \sqrt{\frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2}
    '''
    y_N = np.atleast_1d(y_N)
    yhat_N = np.atleast_1d(yhat_N)
    assert y_N.ndim == 1
    assert y_N.shape == yhat_N.shape
    # sum of squareed error
    sum_squared_error = 0
    for i in range(y_N.shape[0]):
        # sum_squared_error += pow(y_N[i] - yhat_N[i], 2)
        sum_squared_error += np.square(y_N[i] - yhat_N[i])
    mean_squared_error = sum_squared_error / y_N.shape[0]

    return pow(mean_squared_error, 0.5)
    # testing for regrade request 
    # return np.sqrt(np.mean(np.square(y_N - yhat_N)))









####################################################################################################
# This is the 2024S version of this assignment. Please do not remove or make changes to this block.# 
# Otherwise, you submission will be viewed as files copied from other resources.                   # 
####################################################################################################





