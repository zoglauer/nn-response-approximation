## Setup

### Creating the Python environment

Please install the required packages through

```
pip install -r Requirements.txt
```

For anaconda or miniconda users, the sample commands for building a new environment `nn_response` are as follows:

```
conda create --name nn_response python=3.7
conda activate nn_response
pip install -r Requirements.txt
```

This creates the environment, activates it, and installs all required packages.

### Possible Package Issues

These are some possible package issues that could be faced when running code from this branch or other branches.

- ImportError: symbol not found in flat namespace (protobuf 3.20.2)
  \*\* Please use this fix: https://github.com/protocolbuffers/protobuf/issues/10571#issuecomment-1249460270
- No module ‘'mpl_toolkits'
  \*\* Please remove import if this causes an error. Some older branches of this project use this package.
- No module 'tensorflow'
  \*\* M1 macs have had this issue with tensorflow. Please use pip to install tensorflow-macos instead.
- AttributeError: 'FigureCanvasMac' object has no attribute 'set_window_title'
  \*\* Please remove the line if this causes an error. Some older versions of matplotlib have a function that is now deprecated.
- concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.
  \*\* Some older branches do not have code in a **main** method in run.py. Please place it inside a main method.

## Run it

Make sure the environment is activated.

To run the convolutional or fully connected neural network models, please set the desired parameters and run torchrun.py.

To run the denoising autoencoders, please set the desired parameters and run denoiserun.py
