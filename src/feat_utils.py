import numpy as np
from tbox.utils import pkload, pkdump
from functools import partial
import io


def feat_computer():
    return {
            13: {"short": "13-d", "long": "13-dimensional: MFCCs",
             "func": partial(select_compute, n=3, c=f0)},
            39: {"short": "39-d", "long": "39-dimensional: MFCCs + delta + delta-delta",
             "func": partial(select_compute, n=3, c=f0)},
    }


def f0(x):
    return x[np.newaxis, ...]

  
def select_compute(x, n=None, c=None):
    return c(x[:n])


def normalize(X, type, norm_fname):
    """If type is train, compute scaling coefficients, rescale, save the coefficients to file. \
    If type is test, load the coefficients from the training dataset and rescale."""
    mu, sigma = None, None
    if type == "test":
        mu, sigma = pkload(norm_fname)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    if X.ndim == 2:
        X, mu, sigma = _normalize2(X, mu=mu, sigma=sigma)
    elif X.ndim == 3:
        X, mu, sigma = _normalize3(X, mu=mu, sigma=sigma)
    if type == "train":
        pkdump(mu, sigma, fname=norm_fname)
    return X


def _normalize2(X, mu=None, sigma=None):
    """Normalize the columns of X.
    If mu and sigma are provided then use them otherwise compute them."""
    if mu is None and sigma is None:
        mu = X.mean(0)[np.newaxis, :]
        sigma = X.std(0)[np.newaxis, :]
        sigma[sigma == 0] = 1
    return (X-mu)/sigma, mu, sigma


def _normalize3(X, mu=None, sigma=None):
    """Normalize X of size (n_samples, n_feats, n_timeseries).
    If mu and sigma are provided then use them otherwise compute them."""
    if mu is None and sigma is None:
        y = np.swapaxes(X, 1, 2).reshape(-1, X.shape[1])
        mu = y.mean(0)[np.newaxis, :, np.newaxis]
        sigma = y.std(0)[np.newaxis, :, np.newaxis]
        sigma[sigma == 0] = 1

    return (X - mu)/sigma, mu, sigma


def str2double(s):
    try:
        return np.double(s)
    except:
        return np.nan


if __name__ == "__main__":
    pass
