import os
import argparse
from parse import parse
import numpy as np
from tbox.utils import pkload, pkdump
from pre_infectious_detection.src.feat_utils import feat_computer, normalize
from deep_news.src.utils import remove_globalID
from multiprocessing.dummy import Pool


parser = argparse.ArgumentParser(description="Creates several patient data train/test splits.")
parser.add_argument('-opt', metavar='Details string', type=str,
                    help='Example: train-split1-feat1',
                    default="")

parser.add_argument('-exp', metavar='Exp folder', type=str,
                    help='Experiment folders with splits.',
                    default="")

parser.add_argument('-data', metavar='Data folder', type=str,
                    help='Data folder within each split.',
                    default="")


def test_reshape_target():
    y = np.random.rand(2, 1, 2)
    assert(reshape_target(y) == y).all()
    assert(reshape_target(y, alpha=2) == np.array([y[0], y[0], y[1], y[1]])).all()
    assert(reshape_target(y, alpha=3) == np.array([y[0], y[0], y[0], y[1], y[1], y[1]])).all()


def reshape_target(y, alpha=1):
    """Repeat rows alpha times."""
    return np.repeat(y[:, np.newaxis, ...], alpha, axis=1).reshape(-1, y.shape[1], y.shape[2])


if __name__ == "__main__":
    args = parser.parse_args()
    type, isplit, feat_mode = parse("{}-split{}-feat{:d}", args.opt)
    
    exp_folder = args.exp

    data_folder = os.path.join(exp_folder, "split{}".format(isplit), "{}".format(args.data))
    dataset_fname = os.path.join(data_folder, "{}.pkl".format(type))
    norm_fname = os.path.join(data_folder, "train.feat{}.norm.pkl".format(feat_mode))
    gids_fname = os.path.join(data_folder, "train.feat{}.gids.pkl".format(feat_mode))
    out_fname = dataset_fname.replace(".pkl", ".feat{}.pkl".format(feat_mode))

    d, = pkload(dataset_fname)
    d = remove_globalID(d, keep_id=True)
    pkdump(d["gids"], fname=gids_fname)

    c = feat_computer()

    fp = c[feat_mode]["func"]

    #with Pool(2*os.cpu_count()) as pool:
    xp = list(map(fp, d["X"]))
    X = np.concatenate(xp, axis=0)

    alpha = X.shape[1]
    if X.ndim == 3 and alpha == 1 and feat_mode != 0 and feat_mode != 11:
        d["X"] = X.squeeze()
        
    elif X.ndim == 3 and feat_mode != 0 and feat_mode != 11:
        # print("Warning: gids file assume that the number of samples have not been changed.")
        d["X"] = X.reshape(-1, X.shape[2])
        d["Y"] = reshape_target(d["Y"], alpha=alpha)
        
    else:  # feat_mode = 0
        d["X"] = X

    d["X"] = normalize(d["X"], type, norm_fname)
    assert(d["X"].shape[0] == d["Y"].shape[0])
    pkdump(d, fname=out_fname)

