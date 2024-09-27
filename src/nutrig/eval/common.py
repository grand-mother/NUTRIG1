'''
Created on 24 sept. 2024

@author: jcolley
'''

import time
import numpy as np 

quant = 2**13

def get_distrib(model, data):
    nb_bin = 40
    # Load nok data
    t_cpu = time.process_time()
    proba_ok = model.predict(data)
    duration_cpu = time.process_time() - t_cpu
    print(f"Inference CPU time= {duration_cpu} s, for {data.shape[0]} traces")
    hist_ok, bin_edges = np.histogram(proba_ok, nb_bin)
    dist_ok = hist_ok / hist_ok.sum()
    print("sum dist=", dist_ok.sum())
    return dist_ok, bin_edges, proba_ok