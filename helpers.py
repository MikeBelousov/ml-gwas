"""
Custom functions to assisit data processing. 
Authors: Mikhail Belousov, Gennady Khvorykh 
"""

import time
import os, sys
    
def show_time_elepsed(ts):
    # Show time elepsed
    dur = time.time() - ts
    s = time.strftime("%H:%M:%S", time.gmtime(dur))
    print("\nTime elapsed:", s)
    
def check_input(files: list):
    for  file in files:
        if not os.path.isfile(file):
            print(file, "doesn't exist")
            sys.exit(1)