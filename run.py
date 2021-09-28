###################################################################################################
#
# run.py
#
# Copyright (C) by Andreas Zoglauer & contributors
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################

  
  
###################################################################################################

# Base python
import os
import sys
import argparse
import signal

# Own tools
from ToyModel3DConeKeras import ToyModel3DCone
from helpers import *
  
###################################################################################################


"""
This is the main program for the imaging response testing and training in python.
For all the command line options, try:

python3 run.py --help

"""

print("Starting response approximation")


# Parse the command line
parser = argparse.ArgumentParser(description='Perform training and/or testing of the response approximation tool.')
parser.add_argument('-p', '--prefix', default='Run', help='Prefix for saving the output result')
parser.add_argument('-b', '--batch', default=False, action='store_true', help='Run the tool in batch mode, i.e., without UI')


args = parser.parse_args()


AI = ToyModel3DCone()
AI.setBatchMode(args.batch)

if AI.train() == False:
  sys.exit()


# END
###################################################################################################
