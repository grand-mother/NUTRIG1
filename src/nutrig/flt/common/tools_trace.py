'''

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_extra_relatif(extra_rel, dir_angle=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if dir_angle is not None:
        vmin = np.nanmin(dir_angle)
        vmax = np.nanmax(dir_angle)
        norm_user = colors.Normalize(vmin=vmin, vmax=vmax)
        scm = ax.scatter(extra_rel[:, 0], extra_rel[:, 1], extra_rel[:, 2],
                         c=dir_angle, norm=norm_user)
    else:
        scm = ax.scatter(extra_rel[:, 0], extra_rel[:, 1], extra_rel[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    fig.colorbar(scm, label="degree")


def extract_extreltraces_3d(traces3d):
    '''
    Extraction des extremas relatifs sur chaque port
    
    :param traces3: float(nb_trace, 3, idx_sample)
    '''
    nb_trace = traces3d.shape[0]
    size_trace = traces3d.shape[2]
    # extremum absolute, ie > 0
    out_ext_abs = np.empty(nb_trace, dtype=np.float32)
    out_ext_idx = np.empty(nb_trace, dtype=np.int16)
    ext_3d = np.empty(3, dtype=np.float32)
    out_extrel_3d = np.empty((nb_trace, 3), dtype=np.float32)
    for idx_t in range(nb_trace):
        ext_3d_abs = 0.0
        ext_3d_val = 0.0
        ext_3d_port = 0
        ext_3d_idx = 0
        for idx_p in range(3):
            i_max, i_min = 0, 0
            v_max, v_min = 0.0, 0.0
            trace = traces3d[idx_t, idx_p]
            # find extremum in trace
            for idx_s in range(1, size_trace):
                val = trace[idx_s]                
                if val > v_max:
                    i_max = idx_s
                    v_max = val
                elif val < v_min:
                    i_min = idx_s
                    v_min = val
            # with min and max defined the extremum
            abs_min = np.fabs(v_min)
            if abs_min > v_max:
                ext_port = v_min
                ext_port_abs = abs_min
                ext_port_idx = i_min
            else:
                ext_port = v_max
                ext_port_abs = v_max
                ext_port_idx = i_max          
            ext_3d[idx_p] = ext_port
            # check if is the extremum of 3d trace
            if ext_port_abs > ext_3d_abs:
                ext_3d_abs = ext_port_abs
                ext_3d_val = ext_port
                ext_3d_port = idx_p 
                ext_3d_idx = ext_port_idx
        out_ext_abs[idx_t] = ext_3d_abs
        out_ext_idx[idx_t] = ext_3d_idx
        out_extrel_3d[idx_t] = ext_3d / ext_3d_abs
        print(idx_t, ext_3d_abs, ext_3d_port, ext_3d_idx, out_extrel_3d[idx_t])
    return out_extrel_3d, out_ext_abs, out_ext_idx,


class PulsExtractor3D:

    def __init__(self, traces3d):
        '''
        shape (nb_tr, nb_axis=3, nb_s)
        
        :param traces3d:
        '''
        
        t_shape = traces3d.shape
        assert t_shape[1] == 3
        self.traces3d = traces3d
        self.nb_traces = t_shape[0]
        self.size_tr = t_shape[2]
        self.pulse_idx = np.zeros((self.nb_traces, 2), dtype=np.int16)
        # self.traces3d[:, 0, 300] = 200
    
    def extract_fixed_with_extremum(self, before_max=32, after_max=96):
        self.pulses3d = np.empty((self.nb_traces, 3, after_max + before_max), dtype=np.float32)
        for idx in range(self.nb_traces):
            # print(self.traces3d[idx].shape)
            tr_f = self.traces3d[idx].flatten()
            idx_min = np.argmin(tr_f)
            idx_max = np.argmax(tr_f)
            # print(idx_min, idx_max)
            val_min = tr_f[idx_min]
            val_max = tr_f[idx_max]
            if np.abs(val_min) > val_max:
                idx_extm = idx_min % 1024
                # print(idx_extm, idx_min, np.abs(val_min))
            else:
                idx_extm = idx_max % 1024
                # print(idx_extm, idx_max, val_max)
            i_beg = idx_extm - before_max
            i_end = idx_extm + after_max
            if i_beg < 0:
                i_beg = 0
                i_end = after_max + before_max
            elif i_end > self.size_tr:
                i_end = self.size_tr
                i_beg = self.size_tr - (after_max + before_max)                
            self.pulse_idx[idx] = np.array([i_beg, i_end])
            # print(self.pulses3d.shape)
            # print(idx_extm - before_max, idx_extm + after_max)
            self.pulses3d[idx,:,:] = self.traces3d[idx,:, i_beg: i_end ] 
            # print(f"trace {idx} : {self.pulse_idx[idx]}")
            # print(len(tr_f[self.pulse_idx[idx, 0]:self.pulse_idx[idx, 1]]))
            
    def plot_pulse_selected(self, i_beg=3, i_end=6):
        for idx in range(i_beg, i_end):
            plt.figure()
            plt.plot(self.traces3d[idx, 0])
            plt.plot(self.traces3d[idx, 1])
            plt.plot(self.traces3d[idx, 2])
            # plt.plot(np.sum(self.traces3d[idx], axis=0) / 3)
            plt.axvline(self.pulse_idx[idx, 0] , color='b', label='begin')
            plt.axvline(self.pulse_idx[idx, 1] , color='k', label='end')
            plt.legend()
            plt.grid()
            
    def get_pulse3d(self):
        self.extract_fixed_with_extremum(512, 512)
        print(self.pulse_idx.shape, self.pulse_idx.dtype)
        return self.pulses3d
    

class PulsExtractor(object):
    '''    
    Un pulse est un signal a support compact.
    Un pulse a au moins 2 valeurs consécutives hors du bruit
    l'extracteur fournit l'index de début et de fin du pulse avec différentes méthodes
    
    Hypothèses:
      * H1: le pulse est dans la deuxième partie du tableau (> idx_end_noise)
      * H2: la première partie du tableau contient que du bruit
      * H3: le tableau contient un seul pulse
      
      
    Paramètres de l'extracteur:
      * "hors du bruit" : valeur k définissant le seuil relativement à la standard déviation du bruit
      * "dynamique/echantillonnage" : nombres d'échantillon nécessaires 
      
    Auteurs:
      * Jean-Marc Colley CNRS/IN2P3/LPNHE
    '''
    
    def __init__(self, a_trace):
        '''
        
        :param a_trace: array (nb_trace, nb sample in trace)
        '''        
        self.traces = a_trace
        self.nb_trace = a_trace.shape[0]
        self.size_trace = a_trace.shape[1]
        # cf H2 with 4 samples 
        self.idx_end_noise = self.size_trace // 2 - 20
        print(f"{self.nb_trace} trace of {self.size_trace} samples")
    
    def extract_pulse_1(self):
        '''
        with current value and value ahead of 3 samples
        '''
        a_idx_sig = np.zeros((self.nb_trace, 2), dtype=np.int16)
        k_threshold = 4
        std_noise = self.estimate_std_noise()
        mean_trace = self.estimate_mean()
        for idx_t in range(self.nb_trace): 
            trace = self.traces[idx_t].copy() - mean_trace[idx_t]
            print(trace)
            plt.figure()
            plt.plot(trace)
            plt.plot(trace, "*")
            plt.show()
            threshold = k_threshold * std_noise[idx_t]
            print(f"Trace {idx_t}, mean {mean_trace[idx_t]}, threshold {threshold}")
            status = 1
            for idx_s in range(self.idx_end_noise, self.size_trace - 3): 
                t_idx = trace[idx_s]
                dif_v = trace[idx_s + 3] - t_idx
                # print(idx_s, t_idx, dif_v)
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
                                if idx_s - a_idx_sig[idx_t, 0] > 5:
                                    a_idx_sig[idx_t, 1] = idx_s
                                    #
                                    print(f"Fin du signal en {a_idx_sig[idx_t]}")
                                    status = 1
                                    print(idx_s, t_idx, dif_v)
                                else:
                                    status = 1
                                # break
        self.a_idx_sig = a_idx_sig
    
    def extract_pulse_2(self):
        '''
        automate : noise, first pulse, second pulse, fusion first pulse and second pulse
        
        problem: miss sometime pulse with max !
        '''
        # algorithm parameters
        k_threshold = 2.5
        margin_sample = 4
        noise_conf_sample = 5
        calm_ratio = 0.2
        #       
        a_idx_sig = np.zeros((self.nb_trace, 2), dtype=np.int16)
        std_noise = self.estimate_std_noise()
        mean_trace = self.estimate_mean()
        for idx_t in range(self.nb_trace):
            # local copy centred
            trace = self.traces[idx_t].copy() - mean_trace[idx_t]
            print(trace)
            threshold = k_threshold * std_noise[idx_t]
            print(f"Trace {idx_t}, mean {mean_trace[idx_t]}, threshold {threshold}")
            # status : 1 noise, 2 in pulse, 3 conf end pulse
            status = 1
            first_pulse = True
            idx_end = 0
            idx_max_inter_pulse = self.size_trace
            for idx_s in range(self.idx_end_noise, self.size_trace): 
                val = trace[idx_s]
                print(f"{idx_s} {val:.2f}")
                if  (val < -threshold) or (val > threshold):
                    # signal
                    if status == 1:
                        # in pulse now                        
                        idx_begin = idx_s - margin_sample
                        print(f"In pulse {idx_s}, start at {idx_begin}")
                        status = 2
                    elif status == 2:
                        print(f"In pulse {idx_s}")
                        pass
                    elif status == 3:
                        # return "in pulse" status
                        status = 2         
                else:
                    # in noise
                    if status == 1:
                        if not first_pulse and idx_max_inter_pulse == idx_s:
                            # no second pulse
                            print("No second pulse")
                            break 
                    elif status == 2:
                        # end of pulse ?
                        print(f"End of pulse {idx_s} ?")
                        idx_end = idx_s + margin_sample
                        counter_conf = 1
                        status = 3
                    elif status == 3:
                        # may end of pulse
                        counter_conf += 1
                        if counter_conf == noise_conf_sample:
                            # game over
                            print(f"Yes end of pulse {idx_s} at {idx_end}")
                            if first_pulse:
                                print(f"First Pulse")
                                a_idx_sig[idx_t,:] = [idx_begin, idx_end]
                                first_pulse = False
                                # return to "noise" status
                                size_pulse = a_idx_sig[idx_t, 1] - a_idx_sig[idx_t, 0]
                                idx_max_inter_pulse = int(size_pulse * calm_ratio) + idx_end
                                status = 1
                            else:
                                # fusion of pulse ?                                                                
                                dif_idx = idx_begin - a_idx_sig[idx_t, 1]
                                if dif_idx < int(size_pulse * calm_ratio):
                                    # yes fusion
                                    a_idx_sig[idx_t, 1] = idx_end
                                # end of pulse rechearch
                                break
            print(a_idx_sig[idx_t])
            plt.figure()
            plt.plot(trace)
            plt.plot(trace, "*")
            plt.axvline(a_idx_sig[idx_t, 0] , color='b', label='begin')
            plt.axvline(a_idx_sig[idx_t, 1] , color='b', label='begin')
            plt.xlim([450, self.size_trace - 1])            
            plt.show()            
            # processing long pulse        
            if a_idx_sig[idx_t, 1] == 0:
                a_idx_sig[idx_t, 1] = self.size_trace - 1
                            
        self.a_idx_sig = a_idx_sig

    def extract_pulse_3(self):
        '''
        1) collect pulse
            automate : noise => pulse => confirmation out pulse => noise(end)
        2) take pulse with max value and fusion with next if possible
            
        '''
        # algorithm parameters
        k_threshold = 2.5
        margin_sample = 4
        noise_conf_sample = 5
        calm_ratio = 0.2
        #       
        a_idx_sig = np.zeros((self.nb_trace, 2), dtype=np.int16)
        std_noise = self.estimate_std_noise()
        mean_trace = self.estimate_mean()        
        for idx_t in range(self.nb_trace):
            # local copy centred
            trace = self.traces[idx_t].copy() - mean_trace[idx_t]
            print(trace)
            threshold = k_threshold * std_noise[idx_t]
            print(f"Trace {idx_t}, mean {mean_trace[idx_t]}, threshold {threshold}")
            # status : 1 noise, 2 in pulse, 3 conf end pulse
            status = 1
            idx_begin = 0           
            idx_end = 0
            max_val = 0
            l_pulse = []      
            for idx_s in range(self.idx_end_noise, self.size_trace): 
                val = trace[idx_s]
                if val < 0: 
                    val = -val
                print(f"{idx_s} {val:.2f}")
                if  val > threshold:
                    # signal
                    if val > max_val:
                        max_val = val                    
                    if status == 1:
                        # in pulse now                        
                        idx_begin = idx_s - margin_sample
                        print(f"In pulse {idx_s}, start at {idx_begin}")
                        status = 2
                    elif status == 2:
                        print(f"In pulse {idx_s}")
                        pass
                    elif status == 3:
                        # return "in pulse" status
                        print("Confirmation already in signal")
                        status = 2         
                else:
                    # in noise
                    if status == 1:
                        pass
                    elif status == 2:
                        # end of pulse ?
                        print(f"End of pulse {idx_s} ?")
                        idx_end = idx_s + margin_sample
                        counter_conf = 1
                        status = 3
                    elif status == 3:
                        # may end of pulse
                        counter_conf += 1
                        if counter_conf == noise_conf_sample:
                            # game over
                            print(f"Yes end of pulse {idx_s} at {idx_end}")
                            pulse_info = [idx_begin, idx_end, max_val]
                            l_pulse.append(pulse_info)
                            print(f"new pulse {pulse_info}")                                                       
                            # return to "noise" status
                            status = 1                                                    
                            idx_begin = 0           
                            idx_end = 0
                            max_val = 0                                             
            # processing long pulse        
            if idx_end == 0 and idx_begin != 0:
                pulse_info = [idx_begin, self.size_trace - 1, max_val]
                l_pulse.append(pulse_info)
            # define max pulse
            idx_main = 0
            idx_cur = 0
            pulse_main = l_pulse[0]            
            for pulse in l_pulse[1:]:
                print(pulse)
                idx_cur += 1
                if pulse[2] > pulse_main[2]:
                    pulse_main = pulse
                    print(f"new max {pulse_main[2]} at {idx_cur}")
                    idx_main = idx_cur
            print(f"Pulse max {idx_main}: {pulse_main}")
            # fusion pulse
            size_pulse = pulse_main[1] - pulse_main[0] + 1
            max_inter_pulse = int(size_pulse * calm_ratio)
            print(f"max_inter_pulse {max_inter_pulse}")
            if idx_main > 1:
                # prec
                if (pulse_main[0] - l_pulse[idx_main - 1][1]) < max_inter_pulse:
                    pulse_main[0] = l_pulse[idx_main - 1][0]
                    print(f"fusion backward  with {l_pulse[idx_main-1]}")
            for idx_pulse in range(idx_main + 1, len(l_pulse)): 
                if (l_pulse[idx_pulse][0] - pulse_main[1]) < max_inter_pulse:
                    pulse_main[1] = l_pulse[idx_pulse][1]                    
                    print(f"fusion forward  with {l_pulse[idx_pulse]}")
                    print(f"new pulse: {pulse_main}")
                else:
                    break
            print(f"Final Pulse: {pulse_main}")
            print(a_idx_sig[idx_t])
            plt.figure()
            plt.plot(trace)
            plt.grid()
            plt.plot(trace, "*")
            plt.axvline(pulse_main[0], color='b', label='begin')
            plt.axvline(pulse_main[1], color='b', label='begin')
            plt.xlim([450, self.size_trace - 1])
            plt.show()                       
        self.a_idx_sig = a_idx_sig
    
    def estimate_std_noise(self):
        '''
        return array of std deviation for sample [0:self.idx_end_noise]
        '''
        return np.std(self.traces[:, 0:self.idx_end_noise], 1)
    
    def estimate_mean(self):
        '''
        return array of mean value for sample [0:self.idx_end_noise]
        '''
        return np.mean(self.traces[:, 0:self.idx_end_noise], 1)
    
    def estimate_ps_noise(self):
        '''
        return array of power spectrum for sample [0:self.idx_end_noise]
        '''
        # TO DO
        pass

    def wrap_signal(self):
        pass
    
