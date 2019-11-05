import numpy as np
from sklearn.preprocessing import OneHotEncoder


def to_onehot(Y,n=None):
    enc = OneHotEncoder(handle_unknown='ignore')
    if n is None:
        enc.fit(np.unique(Y).reshape(-1, 1))
    else:
        enc.fit(np.arange(n).reshape(-1,1))

    Y_ = enc.transform(Y[:, np.newaxis]).toarray()
    return Y_
