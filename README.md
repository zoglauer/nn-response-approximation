## Setup

### Creating the Python environment
Please install the required packages through
```
pip install -r requirements.txt
```

For anaconda or miniconda users, the sample commands for building a new environment `nn_response` are as follows: 

```
conda create --name nn_response python=3.7
conda activate nn_response
pip install -r requirements.txt
```
This creates the environment, activates it, and installs all required packages.


## Run it
Make sure the environment is activated.
The proof-of-concept approach can be started simply via:
```
python run.py
```
The configurations for training is set in `run.py`.

