from sklearn.metrics import mean_squared_error, accuracy_score,\
                            roc_auc_score, precision_score,\
                            recall_score, average_precision_score
from parse import parse
import numpy as np
from functools import partial
from hq.src.target_utils import to_onehot
from hq.src.utils import githash

eval_metrics = {"MSE": lambda y_true, y_hat: mean_squared_error(y_true, y_hat),
                "Acc": lambda y_true, y_hat: accuracy_score(y_true, cont_to_binary(y_hat,n=y_true.shape[1])),
                "AUROC": lambda y_true, y_hat: roc_auc_score(y_true, y_hat),
                "p": lambda y_true, y_hat: partial(precision_score, average="macro")(y_true, cont_to_binary(y_hat,n=y_true.shape[1])),
                "r": lambda y_true, y_hat: partial(recall_score, average="macro")(y_true, cont_to_binary(y_hat,n=y_true.shape[1])),
                "avgPr": lambda y_true, y_hat: partial(average_precision_score, average="macro")(y_true, y_hat)
                }


def cont_to_binary(y,n=None):
    return to_onehot(np.argmax(y, axis=1),n=n)


def test_cont_to_binary():
    y = np.array([[-1, -2], [-1000, -999], [0.3, 0.7]])
    assert ((cont_to_binary(y) == np.array([[1, 0], [0, 1], [0, 1]])).all())


def test_format_score():
    assert(format_score("abc", 1.23456789) == "abc:1.23457")


def format_score(k, s):
    return "{}:{}".format(k, round(s, 5))


def test_parse_score():
    assert(parse_score("abc:1.23457") == ["abc", 1.23457])


def parse_score(s_str):
    return list(parse("{}:{:f}", s_str))


def format_eval_line(out_mdl, tr_eval_str, te_eval_str):
    return "{} train\t{} test\t{}".format(out_mdl, " ".join(tr_eval_str), " ".join(te_eval_str))


def parse_eval_line(fname):
    with open(fname, "r") as f:
        o = f.read().strip().split("\n")[0]

    _, algo, feat_mode, tr_eval_str, te_eval_str = parse("{}/models/{}.{}.mdl train\t{} test\t{}", o)

    return [algo, feat_mode, [parse_score(s_str) for s_str in tr_eval_str.split(" ")],
                             [parse_score(s_str) for s_str in te_eval_str.split(" ")]]


def write_eval_line(mdl, out_mdl, out_results):
    # Evaluate
    te_eval_str = [format_score(k, fun(mdl.userdata["ttest"], mdl.userdata["xtest_hat"]))
                   for k, fun in eval_metrics.items()]

    tr_eval_str = [format_score(k, fun(mdl.userdata["ttrain"], mdl.userdata["xtrain_hat"]))
                   for k, fun in eval_metrics.items()]
    #
    eval_line = format_eval_line(out_mdl, tr_eval_str, te_eval_str)
    with open(out_results, "w") as f:
        print(githash(), eval_line, file=f)
    return 0


if __name__ == "__main__":
    #test_cont_to_binary()
    pass
