# egamma
Project repository for NBI e/gamma hackathon 2018

### Setup
Different parts of the project require h5py, ROOT, root_numpy, Jupyter notebook, Keras, TensorFlow, and Pytorch. All but the latter can be installed as part of the conda environment specified in [env.yml](env.yml). To set this up, first install e.g. Miniconda2
```bash
$ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
$ bash Miniconda2-latest-Linux-x86_64.sh
$ # Follow the screen prompts
$ # ...
$ rm Miniconda2-latest-Linux-x86_64.sh
```
then create and activate the environment as
```bash
$ conda env create -f env.yml
$ source activate egamma
```
