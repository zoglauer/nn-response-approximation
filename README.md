# Approximating the response of a Compton telescope with a neural network

This toolset aims to approximate the all-sky imaging response of a Compton telescope with a neural network.

For a description of how the imaging response of a Compton telescope looks like, see here: [Zoglauer et a., COSI: From Data to Images](https://arxiv.org/abs/2102.13158)



## Setup


### Prerequisites

The only requirements are access to the command line, git, and Python 3 (3.6 or higher). The following assumes you are using bash either on Windows, Linux, or Mac.


### Get the source code

The current version of the source code is hosted on GitHub. Use these commands to clone the repository:
```
git clone https://github.com/zoglauer/nn-response-approximation ResponseApproximation
```
Then switch to the newly created folder:
```
cd ResponseApproximation
```

### Creating the Python environment

In order to have a clean Python environment into which we can install all the packages we need for training and analysis, we create a virtual python environment. Do this via:

```
python3 -m venv python-env
. python-env/bin/activate
pip3 install -r Requirements.txt
```
This creates the environment, activates it, and installed all required packages.

### Using it

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready to run the proof-of-concept approach.


## Run it

The proof-of-concept approach can be started simply via:
```
python3 run.py
```
This will show two windows, one with the reference cone sections and one with the learned cone sectios.
Press Ctrl-C 3 times to stop the training process.

