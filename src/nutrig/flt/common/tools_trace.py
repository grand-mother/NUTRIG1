'''

'''

import numpy as np
import matplotlib.pyplot as plt


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
        
        @param a_trace: array (nb_trace, nb sample in trace)
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
            plt.plot(trace, "*")
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
    
