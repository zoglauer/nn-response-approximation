###################################################################################################
#
# Helpers.py
#
# Copyright (C) by Andreas Zoglauer & contributors
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################


###################################################################################################


import os
import sys
import signal

###################################################################################################


"""
This sets a fee global variables
"""

# First take care of Ctrl-C
Interrupted = False
NInterrupts = 0


def signal_handler(signal, frame):
    global Interrupted
    global NInterrupts
    print("You pressed Ctrl+C!")
    Interrupted = True
    NInterrupts += 1
    if NInterrupts >= 3:
        print("Aborting!")
        sys.exit()


signal.signal(signal.SIGINT, signal_handler)

# END
###################################################################################################
