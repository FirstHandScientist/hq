import numpy as np
from tbox.ml.vectorizer.var_proc import var_proc
from tbox.utils import pkload, pkdump
from functools import partial
import io
import ctypes
from scipy.stats import skew, kurtosis
from scipy.signal import correlate as xcorr


def feat_computer():
    return {
            0: {"short": "raw VS", "long": "No feature extraction",
            "func": partial(select_compute, n=3, c=f0)},
            1: {"short": "20m mean,std,min,max VS", "long": "20min spo2,BtB,RF extended features",
            "func": partial(select_compute, n=3, c=f1)},
            2: {"short": "20m mean,std,min,max VS+demos", "long": "20min spo2,BtB,RF,BW,Sex,GA extended features",
            "func": partial(select_compute, n=-1, c=f1)},
            3: {"short": "20m mean,std VS", "long": "20min spo2,BtB,RF simple features",
             "func": partial(select_compute, n=3, c=f2)},
            4: {"short": "20m mean,std VS+demos", "long": "20min spo2,BtB,RF,BW,Sex,GA simple features",
             "func": partial(select_compute, n=-1, c=f2)},
            5: {"short": "20m VS AR(15)", "long": "20min spo2,BtB,RF AR(15) coefficients",
             "func": partial(select_compute, n=3, c=partial(f3, p=15))},
            6: {"short": "20m VS AR(30)", "long": "20min spo2,BtB,RF AR(30) coefficients",
             "func": partial(select_compute, n=3, c=partial(f3, p=30))},
            7: {"short": "20m VS AR(60)", "long": "20min spo2,BtB,RF AR(60) coefficients",
             "func": partial(select_compute, n=3, c=partial(f3, p=60))},
            8: {"short": "20m mean,std,BtB_sampen VS", "long": "20min spo2,BtB,RF simple features+BtB_SampEn",
            "func": partial(select_compute, n=3, c=f4)},
            9: {"short": "20m BtB_HRCi", "long": "20min SampAsy, mean, std",
            "func": partial(select_compute, n=3, c=f5)},
            10: {"short": "20m mean,std VS + BtB_HRCi", "long": "20min SampAsy BtB and VS mean, std",
            "func": partial(select_compute, n=3, c=f6)},
            11: {"short": "raw VS, delta-delta", "long": "No feature extraction + delta-deltadelta",
            "func": partial(select_compute, n=3, c=f7)},
            12: {"short": "20m mean,std,skewness,kurtosis xcorr(BtB,SpO2)", "long": " (POPS) 20min spo2,BtB mean,std,skewness,kurtosis cross-correlation.",
            "func": partial(select_compute, n=3, c=f1)},
            13: {"short": "raw VS vectorized", "long": "No feature extraction, vectorize the time series",
            "func": partial(select_compute, n=3, c=f8)},
            14: {"short": "POPS", "long": "mean,std,min,max,kurtosis",
             "func": partial(select_compute, n=3, c=f9)},
    }


def f0(x):
    return x[np.newaxis, ...]

def f1(x):
    return np.array([x.mean(1), x.std(1), x.min(1), x.max(1)]).reshape(1, -1)

def f2(x):
    return np.array([x.mean(1), x.std(1)]).reshape(1, -1)

ARproc = var_proc(lam=0, cv_perc=0)

def f3(x, p=None):
    out = ARproc.solver_(p, x=x)[None, ...]
    assert(not np.isnan(out).any())
    return out

def f4(x):
    return np.concatenate([SampEn(x[1], 4, 0.2), x.mean(1), x.std(1)]).reshape(1, -1)

""" HRC """
def f5(x):
    return np.array([SampAsy(x[1], ref=np.median(x[1])), x[1].mean(), x[1].std()]).reshape(1, -1)

def f6(x):
    return np.concatenate([np.array(SampAsy(x[1], ref=np.median(x[1]))).reshape(1), x.mean(1), x.std(1)]).reshape(1,-1)

def f7(x):
    dx = np.diff(x, axis=1, prepend=np.zeros((3, 1)))
    return np.concatenate((x, dx, np.diff(dx, axis=1, prepend=np.zeros((3, 1)))), axis=0)[np.newaxis, ...]

def f8(x):
    return x.reshape(1, -1)

def f9(x):
    return np.concatenate([x[:2].mean(1), x[:2].std(1), skew(x[:2], axis=1), kurtosis(x[:2], axis=1),
                           max_min_xcorr(x[0], x[1])]).reshape(1, -1)

def max_min_xcorr(x1, x2, delta=30):
    l = x1.shape[0]
    c = xcorr(x1, x2, mode='same')[l // 2 - delta: l // 2 + delta]
    return c.min(), c.max()

def f10(x):
    ddx = f8(x)
    return np.concatenate((x.reshape(1, -1), ddx.reshape(1, -1))).reshape(1, -1)


def SampAsy(x, ref=None):
    d = x - ref
    pos = d > 0
    return (d[pos] ** 2).sum() / (d[~pos] ** 2).sum()


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


def SampEn(U, m, r):
    n  = U.shape[0]
    U_c = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f = io.BytesIO()
   #  with stdout_redirector(f):
       #  _sampen.sampen(U_c, ctypes.c_int(m), ctypes.c_double(r), ctypes.c_int(n))
    #     pass
    out = f.getvalue()
    out_a = np.array([str2double(x.split(" = ")[1]) for x in out.decode('UTF-8').split("\n")[:-1]])
    return out_a


def test_SampEn():
    n = 100
    x = np.arange(100)
    m = 2
    r = 0.21
    SampEn(x, m, r)

if __name__ == "__main__":
    test_SampEn()
    pass
