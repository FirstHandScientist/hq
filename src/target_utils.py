import numpy as np
from sklearn.preprocessing import OneHotEncoder


def to_onehot(Y):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.unique(Y).reshape(-1, 1))
    Y_ = enc.transform(Y[:, np.newaxis]).toarray()
    return Y_
