import numpy as np
from sklearn.preprocessing import OneHotEncoder


def to_onehot(Y):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array([0, 1]).reshape(-1, 1))
    Y_ = enc.transform(Y[:, np.newaxis]).toarray()
    return Y_
