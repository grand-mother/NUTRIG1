import os
import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt

suffixe = "MLP6hybrid/MLP6hybrid_selected.bin"
suffixe = "MLhybrid/MLhybrid_selected.bin"
suffixe = "MLP6/MLP6_selected.bin"
suffixe = "MLP6/MLP6_transient.bin"
# suffixe = "R003562/R003562_A0158_time.bin"


signalfilename = '/sps/trend/slecoz/' + suffixe
# noisefilename = '/sps/trend/slecoz/MLP6SIM/MLP6SIM_transient.bin'

data_title = suffixe.split('/')[-1]

print(data_title)


def read_raw_data(f_name, event_size=1024): 
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


def read_numpy_raw_data(f_name, event_size=1024): 
    return np.fromfile(f_name, np.uint8).reshape((-1, event_size))


def plot_all_event(a_event, data_title):
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


def noise_histo(sig, idx_max=400):
    '''    
    @param sig:
    @param idx_max:
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
    

def plot_interactif():
    # signal = read_raw_data(signalfilename)
    plt.ioff()
    signal = read_numpy_raw_data(signalfilename)
    print(signal.shape)
    x_range = np.arange(400, 600)
    while True:
        nb = int(input("event number: "))
        plt.figure()
        plt.title(f"{data_title} #{nb}") 
        plt.plot(x_range, signal[nb][x_range], label="signal")
        plt.grid()
        plt.show()

    
if __name__ == "__main__":
    signal = read_numpy_raw_data(signalfilename)
    plot_all_event(signal, data_title)
    #noise_histo(signal)
