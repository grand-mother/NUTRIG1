'''
TREND trace is a binary file of unsigned char
Size of the trace is 1024
'''
import numpy as np

SIZE_TRACE = 1024


def read_trace_trend(f_name, event_size=SIZE_TRACE): 
    return np.fromfile(f_name, np.uint8).reshape((-1, event_size))
