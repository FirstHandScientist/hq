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
$ pip install -r requirements.txt
$ echo `realpath ../../../../../` > pyenv/lib/python[3.6]/site-packages
```
The last line adds the folder containing `hq/` and `gm_hmm/` to PYTHONPATH.


