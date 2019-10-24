
import os
import argparse
from deep_news.src.stats import format_stats
import numpy as np
import sys
from hq.src.eval_utils import parse_eval_line

parser = argparse.ArgumentParser(description="This performs simple stats on algorithm results.")


parser.add_argument('-lines', metavar='Option string.', type=str, nargs='+',
                    help='For instance, train-split1-feat1',
                    default=[])

parser.add_argument('-exp', metavar='Exp folder', type=str,
                    help='Experiment folders with splits.',
                    default="")

parser.add_argument('-log', metavar='Log folder', type=str,
                    help='General Log folder.',
                    default="")


if __name__ == "__main__":
    args = parser.parse_args()

    lines = args.lines
    log_folder = args.log
    exp_folder = args.exp

    results = list(map(parse_eval_line, lines))
    res_a = np.array(results)

    algos = np.unique(res_a[:, 0])
    feats = np.unique(res_a[:, 1])

    for algo in algos:
        for feat in feats:
            idx = np.bitwise_and(res_a[:, 0] == algo, res_a[:, 1] == feat)
            id_str = "{}.{}".format(algo, feat)
            outfile = os.path.join(exp_folder, log_folder, id_str + ".Acc")
            tr_score = np.array([x for x in res_a[idx, 2]])
            te_score = np.array([x for x in res_a[idx, 3]])

            tr_scores = {tr_score[0, i, 0]: list(map(float, tr_score[:, i, 1].tolist())) for i in
                         range(tr_score.shape[1])}
            te_scores = {te_score[0, i, 0]: list(map(float, te_score[:, i, 1].tolist())) for i in
                         range(te_score.shape[1])}

            with open(outfile, "w") as f:
                print("{} train:{}".format(id_str,
                                            " ".join(["{}:{}".format(k, format_stats(v)) for k,v in tr_scores.items()]))
                , file=f)
                print("{} test:{}".format(
                                            id_str,
                                            " ".join(["{}:{}".format(k, format_stats(v)) for k,v in te_scores.items()]))
                , file=f)

    sys.exit(0)
