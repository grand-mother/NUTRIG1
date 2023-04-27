'''
Created on 25 avr. 2023

@author: jcolley
'''

import pprint

import numpy as np

import matplotlib.pylab as plt

import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent
import sradio.io.sradio_asdf as fsr
from sradio.num.signal import WienerDeconvolution
import sradio.model.ant_resp as ant
from  sradio.basis.frame import FrameDuFrameTan

#
# logger
#
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("debug")


#
# FILES
#


FILE_voc = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out_v_oc.asdf"
PATH_leff = "/home/jcolley/projet/grand_wk/data/model/detector"

def get_simu_magnetic_vector(d_simu):
    d_inc = d_simu["geo_mag2"]["inc"]
    r_inc = np.deg2rad(d_inc)
    v_b = np.array([np.cos(r_inc), 0, -np.sin(r_inc)])
    logger.info(f"Vec B: {v_b} , inc: {d_inc:.2f} deg")
    return v_b

def get_simu_xmax(d_simu):
    xmax = 1000.0 * np.array(
        [d_simu["x_max"]["x"], d_simu["x_max"]["y"], d_simu["x_max"]["z"]]
    )
    return xmax

def check_recons_with_white_noise():
    """
    1) read v_oc file
    2) create wiener object
    3) on trace
        * add white noise
        * compute relative xmax and direction
        * compute polarization angle
        * get Leff for direction and polar angle
        * deconv and store
    4) plot result
    5) estimate polarization or B orthogonality for all traces
    """
    # 1)
    evt, d_simu = fsr.load_asdf(FILE_voc)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    # 2) 
    wiener = WienerDeconvolution(evt.f_samp_mhz*1e-6)
    # 3) 
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files(PATH_leff))
    #evt.plot_footprint_val_max()
    idx_du = 44
    evt.plot_trace_idx(idx_du)
    ##add white noise
    sigma = 10
    noise = np.random.normal(0, sigma, (3,evt.get_size_trace()))
    evt.traces[idx_du] += noise
    evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    ant3d.set_name_pos(evt.du_id[idx_du], evt.network.du_pos[idx_du])
    ## compute polarization angle
    v_b = get_simu_magnetic_vector(d_simu)
    v_pol = np.cross(ant3d.cart_src_du, v_b)
    v_pol /= np.linalg.norm(v_pol)
    logger.info(f"vec pol: {v_pol}")
    assert np.allclose(np.dot(v_pol, v_b), 0)
    tandu = FrameDuFrameTan(ant3d.dir_src_du)
    v_pol_tan= tandu.vec_to_b(v_pol)
    print(v_pol_tan)
    v_pol_tan= tandu.vec_to_b(ant3d.cart_src_du)
    print(v_pol_tan)
    ## get Leff for direction
    
    
    

if __name__ == '__main__':
    check_recons_with_white_noise()
    plt.show()