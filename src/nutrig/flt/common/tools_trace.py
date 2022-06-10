'''

'''

import numpy as np
import matplotlib.pyplot as plt

class DefineSignalInTrace(object):
    '''
    trace contains a signal with noise before and after
    '''
    
    def __init__(self, a_trace):
        '''
        
        @param a_trace: array (nb_trace, nb sample in trace)
        '''
        self.idx_end_noise = 500
        self.traces = a_trace
        self.nb_trace = a_trace.shape[0]
        self.size_trace = a_trace.shape[1]
        print(f"{self.nb_trace} trace of {self.size_trace} samples")
    
    def extract_signal(self):
        '''
        find begin and end idx of the signal in trace
        '''
        a_idx_sig = np.zeros((self.nb_trace, 2), dtype=np.int16)
        k_threshold = 4
        std_noise = self.estimate_std_noise()
        mean_trace = self.estimate_mean_noise()
        for idx_t in range(self.nb_trace): 
            trace = self.traces[idx_t].copy() - mean_trace[idx_t]
            print(trace)
            plt.figure()
            plt.plot(trace)
            plt.plot(trace,"*")
            plt.show()
            threshold = k_threshold * std_noise[idx_t]
            print(f"Trace {idx_t}, mean {mean_trace[idx_t]}, threshold {threshold}")
            status = 1
            for idx_s in range(self.idx_end_noise, self.size_trace - 3):                
                t_idx = trace[idx_s]
                dif_v = trace[idx_s + 3] - t_idx
                #print(idx_s, t_idx, dif_v)
                if status == 1:
                    # in noise                    
                    if dif_v > threshold:
                        status = 2  # increasing
                        a_idx_sig[idx_t, 0] = idx_s
                    elif dif_v < -threshold:
                        status = 2  # decreasing
                        a_idx_sig[idx_t, 0] = idx_s
                elif status == 2:
                    # in signal
                        if -threshold < dif_v < threshold:
                            if -threshold < t_idx < threshold:
                                # in noise
                                if idx_s - a_idx_sig[idx_t, 0] > 5 :
                                    a_idx_sig[idx_t, 1] = idx_s
                                    #
                                    print(f"Fin du signal en {a_idx_sig[idx_t]}")
                                    status = 1
                                    print(idx_s, t_idx, dif_v)
                                else:
                                    status = 1
                                #break
        self.a_idx_sig = a_idx_sig
    
    def estimate_std_noise(self):
        '''
        return array std deviation for sample [0:self.idx_end_noise]
        '''
        return np.std(self.traces[:, 0:self.idx_end_noise], 1)
    
    def estimate_mean_noise(self):
        '''
        return array std deviation for sample [0:self.idx_end_noise]
        '''
        return np.mean(self.traces[:, 0:self.idx_end_noise], 1)
    
    def estimate_ps_noise(self):
        '''
        return array std deviation for sample [0:self.idx_end_noise]
        '''
        pass

    def wrap_signal(self):
        pass
    
