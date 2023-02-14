import os
import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt

import nutrig.data_api.du.trend.io as tio

# CCIN2P3
PATH_DATA = "/sps/trend/slecoz/"
# local
PATH_DATA = "/home/jcolley/projet/nutrig_wk/data/"



def read_trend_ref(f_name, event_size=1024): 
    filesize = os.path.getsize(f_name)
    nb_ev = int(filesize / event_size)
    signal = np.zeros((nb_ev, event_size))
    # signal = np.empty((nb_ev, event_size), dtype=np.uint8)
    idx_ev = 0
    with open(signalfilename, 'rb') as fd:
        while idx_ev < nb_ev:
            content = fd.read(event_size)
            signal[idx_ev] = struct.unpack('B' * event_size, content)
            idx_ev += 1
    return signal



def plot_all_event(a_event, data_title=""):
    matplotlib.use('Agg')
    nb_event = a_event.shape[0]
    x_range = np.arange(400, 600)
    for idx_ev in range(nb_event):
        print(f'{idx_ev}/{nb_event}')
        plt.figure(0)
        plt.ylim([-1, 300])
        plt.title(f"{data_title} #{idx_ev}")
        plt.plot(x_range, signal[idx_ev][x_range])
        plt.grid()
        plt.savefig(f"event/gerbe_{idx_ev:05}.png")
        plt.close(0)


def noise_histo(sig, data_title="", idx_max=400):
    '''    
    :param sig:
    :param idx_max:
    '''
    a_mean = np.mean(sig[:, 0:idx_max], 1)
    a_std = np.std(sig[:, 0:idx_max], 1)
    print(a_std.shape)
    print(np.min(a_std), np.max(a_std))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"noise between [0:{idx_max}], file {data_title}")
    ax1.set_title("histo mean")
    ax1.hist(a_mean)
    ax1.set_xlabel("ADU")
    ax1.grid()
    ax2.set_title("histo std deviation")
    ax2.hist(a_std)
    ax2.set_xlabel("ADU")
    ax2.grid()
    plt.show()
    

def plot_interactif(signal, data_title=""):
    # signal = read_raw_data(signalfilename)
    plt.ion()
    print(signal.shape)
    x_range = np.arange(400, 800)
    while True:
        nb = int(input("event number: "))
        plt.figure(nb)
        plt.title(f"{data_title} #{nb}") 
        plt.plot(x_range, signal[nb][x_range], label="signal")
        plt.grid()
        plt.show()

    
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
    plot_interactif(signal, data_title)
    #noise_histo(signal)
