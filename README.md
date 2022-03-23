# postmit

Python package to do some post-processing on model ouput from the [MITgcm](https://mitgcm.readthedocs.io/en/latest/) to calculate variables not included in the model output etc.   

The package works based on output contained in [`xarray`](https://docs.xarray.dev/en/stable/) datasets, i.e. loaded with [`xmitgcm`](https://xmitgcm.readthedocs.io/en/latest/) or similarly.   


## Install a minimal environment with conda

~~~bash
# clone the git repository
git clone git@github.com:jk-rieck/postmit.git
cd postmit
# create and activate the environment
conda env create -f environment.yml
conda activate py3_postmit
# install eddytools
pip install -e .
~~~


## Install in existing environment

1. Make sure you have `numpy`, `xarray`, `xgcm`, `gsw` and `MITgcmutils` installed.

2. Install from the repository using
  ~~~bash
  pip install git+https://github.com/jk-rieck/postmit.git@main
  ~~~

## Usage

See the [__example notebook__](https://github.com/jk-rieck/postmit/blob/main/examples/example.ipynb).
