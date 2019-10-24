import numpy as np
from hq.src.utils import pkload, pkdump
from functools import partial
import os
import json
import sys


def feat_computer():
    return {
            0: {"short": "Identity", "long": "Identity",
             "func": lambda x: x}
    }

def norm_minmax(x, min_=None, max_=None):
    return ((x - min_.reshape(1, -1)) / (max_.reshape(1, -1) - min_.reshape(1, -1)))

def normalize(xtrain, xtest):
    """Normalize training data set between 0 and 1. Perform the same scaling on the testing set."""
    f_min = np.vectorize(lambda x: np.min(x, axis=0), signature="()->(k)")
    f_max = np.vectorize(lambda x: np.max(x, axis=0), signature="()->(k)")
    min_tr = np.min(f_min(xtrain), axis=0)
    max_tr = np.max(f_max(xtrain), axis=0)

    # The first component is zeros and can create division by 0
    min_tr[0] = 0
    max_tr[0] = 1
    f_perform_normalize = np.vectorize(partial(norm_minmax, min_=min_tr, max_=max_tr), signature="()->()", otypes=[np.ndarray])
    return f_perform_normalize(xtrain), f_perform_normalize(xtest)


def getsubset(data, label, iphn):
    # concat data
    # find subset
    idx = np.in1d(label, iphn)
    return data[idx], label[idx]


def find_change_loc(x):
    dx = np.diff(x)
    # Make a clean vector to delimit phonemes
    change_locations = np.array([0] + (1 + np.argwhere(dx != 0)).reshape(-1).tolist() + [x.shape[0]])
    # Make an array of size n_phoneme_in_sentence x 2, containing begining and end of each phoneme in a sentence

    fmt_interv = np.array([[change_locations[i-1], change_locations[i]]\
                                 for i in range(1, change_locations.shape[0]) ])
    return fmt_interv, x[change_locations[:-1]]


def test_find_change_loc():
    l = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,3])
    out, out2 = find_change_loc(l)
    assert((out2 == np.array([1,0,2,3])).all())
    assert((out == np.array([[0,  7], [7, 13],[13, 19],[19, 20]])).all())

    l = np.array([1, 1, 0, 0, 2, 2])
    out, out2 = find_change_loc(l)
    assert((out2 == np.array([1, 0, 2])).all())
    assert((out == np.array([[0, 2], [2, 4], [4, 6]])).all())


def to_phoneme_level(DATA):
    n_sequences = len(DATA)

    seq_train = [0 for _ in range(n_sequences)]
    targets_train = [0 for _ in range(n_sequences)]
    data_tr = []
    labels_tr = []

    # For all sentences
    for i, x in enumerate(DATA):
        seq_train[i], targets_train[i] = find_change_loc(x[:, 0])

        # Delete label from data
        x = np.delete(x, 0, axis=1)
        #  x[:, 0] = 0

        # For each phoneme found in the sentence, get the sequence of MFCCs and the label
        for j in range(seq_train[i].shape[0]):
            data_tr += [x[seq_train[i][j][0]:seq_train[i][j][1]]]
            labels_tr += [targets_train[i][j]]

    # Return an array of arrays for the data, and an array of float for the labels
    return np.array(data_tr), np.array(labels_tr)

def remove_label(data, labels, phn2int_39):
    keep_idx = labels != phn2int_39['-']
    data_out = data[keep_idx]
    label_out = labels[keep_idx]
    assert(len(label_out) == data_out.shape[0])
    return data_out, label_out


def phn61_to_phn39(label_int_61, int2phn_61=None, data_folder=None, phn2int_39=None):
    """Group labels based on info found on table 3 of html file."""
    with open(os.path.join(data_folder, "phoneme_map_61_to_39.json"), "r") as fp:
        phn61_to_39_map = json.load(fp)

    label_str_61 = [int2phn_61[int(x)] for x in label_int_61]

    label_str_39 = [phn61_to_39_map[x] if x in phn61_to_39_map.keys() else x for x in label_str_61 ]

    # At this point there is still 40 different phones, but '-' will be deleted later.
    if phn2int_39 is None:
        unique_str_39 = list(set(label_str_39))
        phn2int_39 = {k: v for k, v in zip(unique_str_39, range(len(unique_str_39)))}

    label_int_39 = [phn2int_39[x] for x in label_str_39]
    return np.array(label_int_39), phn2int_39


def test_flip():
    d = {k: v for k, v in zip(list("abcbdefg"), list(range(8)))}
    assert(d == flip(flip(d)))

def flip(d):
    """In a dictionary, swap keys and values"""
    return {v: k for k, v in d.items()}

def read_classmap(folder):
    fname = os.path.join(folder, "class_map.json")
    if os.path.isfile(fname):
        with open(fname, "r") as f:
            return json.load(f)
    else:
        return {}


def write_classmap(class2phn, folder):
    """Write dictionary to a JSON file."""
    with open(os.path.join(folder, "class_map.json"), "w") as outfile:
        out_str = json.dumps(class2phn, indent=2)
        print("Classes are: \n" + out_str, file=sys.stderr)
        outfile.write(out_str+"\n")
    return 0


def str2double(s):
    try:
        return np.double(s)
    except:
        return np.nan


if __name__ == "__main__":
    pass
