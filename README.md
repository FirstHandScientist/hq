# Install
## dependencies
```bash
$ git clone https://gitlab.com/antoinehonore/hq.git
$ git clone -b api https://gitlab.com/antoinehonore/gm_hmm.git
```
## Interpreters
In `hq/` and `gm_hmm/` folders
Run:
```bash
$ cd [FOLDER]
$ virtualenv -p python3 pyenv
$ . pyenv/bin/activate
$ python [FOLDER]/setup.py develop
```
The last line adds the folder containing `hq/` and `gm_hmm/` to PYTHONPATH.


# Data
Place the train and test pickle files corresponding to 61 classes and 39 features and clean test with the names: `train.feat0.pkl` and `test.feat0.pkl` under `exp/split\_c61f39clean/data`.
Follow the same naming notations for the rest of the possible datasets.
- feat0 is not used here but might be necessary if we were to compute different sets of features.

# Test
From `hq/`, run:
```bash
$ make model=gmmhmm splits=_c61f39clean feats=0
```

# Advanced test
Once the previous test runs and gives results, we can try more advanced calls:
```bash
$ ./gmmhmm_submodels
$ ./gmmhmm_submodels.sh print
Models:  gmmhmm
Number of states (ns):  3 6 9
Number of iterations (niter):  2 10 20
Number of mixtures (nmix):  2 4 6 8 10 12
gmmhmm-ns\{3,6,9\}-niter\{2,10,20\}-nmix\{2,4,6,8,10,12\}
```
Copy the last line and use it in the make call
```bash
$ make gmmhmm-ns\{3,6,9\}-niter\{2,10,20\}-nmix\{2,4,6,8,10,12\} splits=_c61f39clean feats=0 -j 5
```

This allows to train all combinations of hyper parameters for gmmhmms on the data called c61f39clean.
Once you have set up more split_ folders, you can run things like:

```bash
$ make gmmhmm-ns\{3,6,9\}-niter\{2,10,20\}-nmix\{2,4,6,8,10,12\} splits=_c61f\{39,13\}clean feats=0 -j 5
```


# Acknowledge
This repos is forked from [hq](https://gitlab.com/antoinehonore/hq/tree/master)


