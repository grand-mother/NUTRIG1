"""
Created on 25 avr. 2023

@author: jcolley
"""

import pprint

import numpy as np
import scipy.fft as sf
import scipy.optimize as sco
import matplotlib.pylab as plt

import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent
import sradio.io.sradio_asdf as fsr
from sradio.num.signal import WienerDeconvolution
import sradio.model.ant_resp as ant
from sradio.basis.frame import FrameDuFrameTan
from sradio.basis import coord
from sradio.num.signal import get_fastest_size_rfft

#
# logger
#
logger = mlg.get_logger_for_script("script")
mlg.create_output_for_logger("debug", log_root="script")


#
# FILES
#
FILE_voc = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out_v_oc.asdf"
PATH_leff = "/home/jcolley/projet/grand_wk/data/model/detector"




if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # 
    pass
    # 
    logger.info(mlg.string_end_script())
    plt.show()
