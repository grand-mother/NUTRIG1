"""
use ASDF 

https://asdf.readthedocs.io/en/stable/index.html

"""

import os.path
from logging import getLogger

import numpy as np
import asdf

from sradio.basis.traces_event import Handling3dTracesOfEvent


logger = getLogger(__name__)


class ZhairesSingleEventAsdf:
    def __init__(self):
        self.d_zh = None

    def __call__(self, path_zhaires):
        try:
            self.d_zh.close()
        except:
            pass
        self.d_zh = asdf.open(path_zhaires)
        return self.d_zh

    def __del__(self):
        try:
            self.d_zh.close()
        except:
            pass

    def get_object_3dtraces(self):
        o_tevent = Handling3dTracesOfEvent(f"ZHAIRES simulation")
        nb_ant = self.d_zh["ant_pos"].shape[0]
        du_id = np.arange(nb_ant)
        #  MHz/ns: 1e-6/1e-9 = 1e3
        sampling_freq_mhz = 1e3 / self.d_zh["t_sample_ns"]
        o_tevent.init_traces(
            self.traces,
            du_id,
            self.t_start,
            sampling_freq_mhz,
        )
        ants = np.empty((nb_ant, 3), dtype=np.float32)
        ants[:, 0] = self.ants["x"]
        ants[:, 1] = self.ants["y"]
        ants[:, 2] = self.ants["z"]
        o_tevent.init_network(self.ants)
        o_tevent.set_unit_axis("$\mu$V/m", "cart")
        return o_tevent
