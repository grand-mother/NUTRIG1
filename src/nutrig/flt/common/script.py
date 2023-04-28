'''


'''

import matplotlib.pyplot as plt

import nutrig.flt.common.tools_trace as tt
import nutrig.data_api.du.trend.io as tio

# CCIN2P3
PATH_DATA = "/sps/trend/slecoz/"
# local
PATH_DATA = "/home/jcolley/projet/nutrig_wk/data/"

if __name__ == "__main__":
    # file 
    file_trace = "MLP6hybrid/MLP6hybrid_selected.bin"
    file_trace = "MLP6/MLP6_selected.bin"
    file_trace = "MLP6/MLP6_transient.bin"
    file_trace = "MLhybrid_selected.bin"
    signalfilename = PATH_DATA + file_trace
    data_title = signalfilename.split('/')[-1]
    print(data_title)
    # 
    signal = tio.read_trace_trend(signalfilename)
    def_sig = tt.PulsExtractor(signal[10:20,:])
    
    ret = def_sig.extract_pulse_1()
    print(ret)
    # noise_histo(signal)
