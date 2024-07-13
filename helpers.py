"""
Custom functions to assisit data processing. 
Authors: Mikhail Belousov, Gennady Khvorykh 
"""

import time
    
def show_time_elepsed(ts):
    # Show time elepsed
    dur = time.time() - ts
    s = time.strftime("%H:%M:%S", time.gmtime(dur))
    print("\nTime elapsed:", s)
    
