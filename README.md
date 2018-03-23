# egamma
Project repository for NBI e/gamma hackathon 2018

### Setup

Different parts of the project require h5py, ROOT, root_numpy, Jupyter notebook,
Keras, TensorFlow, and Pytorch. These can all be installed as part of the conda
environments specified in [envs/](envs/). To set this up, first install
e.g. Miniconda2
```bash
$ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
$ bash Miniconda2-latest-Linux-x86_64.sh
$ # Follow the screen prompts
$ # ...
$ rm Miniconda2-latest-Linux-x86_64.sh
```
then create and activate e.g. the CPU environment as
```bash
$ conda env create -f envs/egamma-cpu.yml
$ source activate egamma-cpu
```
Similarly, [envs/egamma-gpu.yml](envs/egamma-gpu.yml) provide an environment
suitable for running on GPU(s).


### Known issues

For some reason, there is a problem with linking to libstdc++ whe importing
TensorFlow, manifesting as an error like:
```bash
ImportError: <conda/env/dir>/lib/./libstdc++.so.6:
version `CXXABI_1.3.9' not found (required by <conda/env/dir>/lib/python2.7/site-packages/google/protobuf/pyext/_message.so)
```
This can be fixed as follow:
```bash
$ source activate egamma
$ envdir="$(conda info --env | grep \* | sed 's/.* //g')"
$ linkpath="$envdir/lib/libstdc++.so.6"
$ latestlib="$(find $envdir/lib/ -name libstdc++.* ! -type l | grep -v .py | sort | tail -1)"
$ ln -s -f $latestlib $linkpath
```
