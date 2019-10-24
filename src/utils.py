import os
from glob import glob
import numpy as np
#from tbox.utils import pkload
from parse import parse
import argparse
from hq.src.target_utils import to_onehot
import pickle


def pkload(fname="a.out"):
    out = pickle.load(open(fname, "rb"))
    return out


def pkdump(*args, fname="a.out"):
    pickle.dump(args, open(fname, "wb"))
    return 0


def test_get_algoname():
    assert(get_algoname("fmds.kflnsdf/lsdjgbkfhlasd/ldskja/run_myalgo.py") == "myalgo")


def get_algoname(fname):
    return parse("run_{}.py", os.path.basename(fname))[0]


def test_getclass():
    assert(getclass("udbfuisd/hsdbfkasdbfk/jkfsdlkf/featmskf_class12.pkl") == 12)


def getclass(fname):
    return parse("{}_class{:d}.pkl", fname)[1]


def gather_data(type, feat_mode, folder_):
    inclassfiles = fetch_class_files(os.path.join(folder_, "{}.feat{}_class*.pkl".format(type, feat_mode)))
    gidsclassfiles = fetch_class_files(os.path.join(folder_, "{}.feat{}.gids_class*.pkl".format(type, feat_mode)))

    class_ = list(map(getclass, inclassfiles))
    x_list = list(map(lambda x: np.array(pkload(x)[0]), inclassfiles))
    gids_list = list(map(lambda x: np.array(pkload(x)[0]), gidsclassfiles))
    Y = np.array(sum([[class_[i] - 1 for _ in range(x_list[i].shape[0])] for i, fname in enumerate(inclassfiles)], []))
    X = np.concatenate([a for a in x_list if not a.shape[0] == 0], axis=0)
    gids = np.concatenate(gids_list)
    Y = to_onehot(Y)
    return X, Y, gids


def fetch_class_files(pattern):
    get_sort_key = lambda x: parse("{}_class{:d}.pkl", x)[1]
    return sorted(glob(pattern), key=get_sort_key)


def gather_data_gen(type, feat_mode, folder_):
    inclassfiles = fetch_class_files(os.path.join(folder_, "{}.feat{}_class*.pkl".format(type, feat_mode)))
    gidsclassfiles = fetch_class_files(os.path.join(folder_, "{}.feat{}.gids_class*.pkl".format(type, feat_mode)))

    class_ = list(map(getclass, inclassfiles))
    x_list = list(map(lambda x: np.array(pkload(x)[0]).swapaxes(1,2), inclassfiles))
    gids_list = list(map(lambda x: np.array(pkload(x)[0]), gidsclassfiles))
    y_list = [[class_[i] for _ in range(x_list[i].shape[0])] for i, fname in enumerate(inclassfiles)]
    gids = gids_list
    return x_list, y_list, gids


def run_parser(desc=""):
    parser = argparse.ArgumentParser(description="Performs {}.".format(desc))

    parser.add_argument('-opt', metavar='Option string.', type=str,
                        help='For instance, train-split1-feat1',
                        default="")

    parser.add_argument('-exp', metavar='Exp folder', type=str,
                        help='Experiment folders with splits.',
                        default="")

    parser.add_argument('-data', metavar='Data folder', type=str,
                        help='Data folder within each split.',
                        default="")

    parser.add_argument('-mdl', metavar='Model folder', type=str,
                        help='Model folder within each split.',
                        default="")

    parser.add_argument('-log', metavar='Log folder', type=str,
                        help='Log folder within each split.',
                        default="")
    return parser

