import matplotlib.pyplot as plt
import numpy as np
import sklearn
from functools import partial
from multiprocessing.dummy import Pool
from tbox.utils import pkload, githash
import os
from parse import parse


def plot_learning_curve(title, train_sizes, train_scores, test_scores, ylim=None, save=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    train_scores_mean = train_scores.mean(0)
    train_scores_std = train_scores.std(0)
    test_scores_mean = test_scores.mean(0)
    test_scores_std = test_scores.std(0)

    plt.figure(frameon=False)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training Patients")
    plt.ylabel("Accuracy")

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")
    plt.legend(loc="best")
    if not (save is None):
        plt.savefig(save)
    plt.close()
    return 0


def run_lc(model, x, y, xtest=None, ytest=None):
    model.fit(x, y)
    tr_ = model.score(x, y)
    te_ = model.score(xtest, ytest)
    return tr_, te_


def get_lc_scores(estimator, X, Y, gids, xtest, ytest):
    ids, npat, train_sizes = get_lc_indexes(gids, Y, step=1)

    fun = partial(run_lc, xtest=xtest, ytest=ytest)

    estimators = [sklearn.base.clone(estimator) for _ in range(npat)]

    Xs = [X[i] for i in ids]
    Ys = [Y[i] for i in ids]

    i = 0

    #for a, b, c in zip(estimators, Xs, Ys):
    #    print("here", i)
    #    out = fun(a, b, c)
    #    i = i+1


    with Pool(len(estimators)) as pool:
        out = pool.starmap(fun, zip(estimators, Xs, Ys))

    train_scores = [x[0] for x in out]
    test_scores = [x[1] for x in out]
    return train_sizes, train_scores, test_scores


def get_lc_indexes(gids, Y_, step=1):
    """Builds a list of boolean indexes selecting samples from a growing number of patients.
    TODO: make multiclass management better."""

    if Y_.ndim == 1:
        Y = Y_[:, None]
    else:
        Y = Y_

    npat = np.unique(gids).shape[0]
    train_sizes = np.arange(1, npat, step) + 1

    # Make sure that the first patient has positive samples
    uniq_ids = np.unique(gids)[np.random.permutation(npat)]

    first_ones = np.argwhere([(Y[gids == i, 0] == 1).sum() > 0 for i in uniq_ids]).reshape(-1)[0]
    if first_ones > 0:
        uniq_ids[0], uniq_ids[first_ones] = uniq_ids[first_ones], uniq_ids[0]

    first_zeros = np.argwhere([(Y[gids == i, 0] == 0).sum() > 0 for i in uniq_ids]).reshape(-1)[0]
    if first_zeros > 1:
        uniq_ids[1], uniq_ids[first_zeros] = uniq_ids[first_zeros], uniq_ids[1]

    ids = [np.in1d(gids, uniq_ids[:i]) for i in train_sizes]
    assert (np.unique(Y[ids[0]][:, 0]).shape[0] > 1)
    return ids, npat, train_sizes


def gen_graph(consistent_lines,exp_folder=None,log_folder=None,feats_detail=None):
    """Plot the LC averaged from experiments detailed in line."""
    algoname, feat_mode = parse("{}.feat{:d}.lc", os.path.basename(consistent_lines[0]))
    out_fig = os.path.join(exp_folder, log_folder, "{}.feat{}.lc.pdf".format(algoname, feat_mode))
    title_str = "{}: {}, {}".format(githash(), algoname, feats_detail[feat_mode]["short"])

    results = list(map(pkload, consistent_lines))

    # Sometimes all training sets accross splits do not have exactly the same number of training patient.
    l = min([len(r[0]) for r in results])
    assert (all([results[0][0][:l].shape == r[0][:l].shape for r in results]))

    train_sizes = results[0][0][:l]
    train_scores = np.concatenate([np.array(r[1])[np.newaxis, :l] for r in results], axis=0)
    test_scores = np.concatenate([np.array(r[2])[np.newaxis, :l] for r in results], axis=0)

    plot_learning_curve(title_str, train_sizes, train_scores, test_scores, save=out_fig)
    return 0
