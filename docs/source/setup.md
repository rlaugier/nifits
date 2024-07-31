(setup_tips)=

# Installing the nifits package

This procedure was mostly tested on ubuntu. If problems accur attempting to install on other platforms, please file an issue on the github repository.

## Dependencies

Although initially developped for python 3.8 and numpy 1.24, the setup installation routine was not working reliably.

After some patches (version 0.0.3+), we now recommend python 3.12 and numpy 2.0+ for a smooth experience. Most dependencies should install automatically. 

The main dependencies are:

* astropy
* numpy
* scipy
* matplotlib
* sympy

## setup

Working in a specific conda environment is recommended.

Clone the repo:

```bash
git clone https://github.com/rlaugier/nifits
```

Navigate to the repo and setup:

```bash
cd nifits
python setup.py develop
```

Note that using the action `develop` is recommended during this period of active development to avoid the need to re-install each time you pull the repo. Just restarting the kernel should suffice, unless there are changes in the package architecture.