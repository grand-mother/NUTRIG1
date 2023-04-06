from logging import getLogger

import h5py
import numpy as np

from sradio.basis.traces_event import Handling3dTracesOfEvent

logger = getLogger(__name__)


class ZhairesSingleEventHdf5:
    def __init__(self, path_hdf5):
        self.d_zh = None
        f_zh = h5py.File(path_hdf5)
        name_data = f_zh["RunInfo"]["EventName"][0]
        self.data = f_zh[name_data]

    def _get_traces(self):
        self.ants_id = list(self.data["AntennaTraces"])
        self.d_ants_idx = {a_id: idx for idx, a_id in enumerate(self.ants_id)}
        self.nb_ants = len(self.ants_id)
        size_trace = self.data["AntennaTraces"][self.ants_id[0]]["efield"].shape[0]
        self.traces = np.empty((self.nb_ants, 3, size_trace), dtype=np.float32)
        self.t_start = np.empty(self.nb_ants, dtype=np.float64)
        for idx, a_id in enumerate(self.ants_id):
            trace = self.data["AntennaTraces"][a_id]["efield"]
            self.traces[idx][0] = trace["Ex"]
            self.traces[idx][1] = trace["Ey"]
            self.traces[idx][2] = trace["Ez"]
            self.t_start[idx] = trace["Time"][0]
        self.t_sample_ns = trace["Time"][1] - trace["Time"][0]
        print(self.t_sample_ns)

    def _get_antspos(self):
        self.ants_pos = np.empty((self.nb_ants, 3), dtype=np.float32)
        for idx in range(self.nb_ants):
            aid = self.data["AntennaInfo"][idx]["ID"]
            a_idx = self.d_ants_idx[str(aid, "UTF-8")]
            ant_pos = self.data["AntennaInfo"][idx]
            self.ants_pos[a_idx][0] = ant_pos["X"]
            self.ants_pos[a_idx][1] = ant_pos["Y"]
            self.ants_pos[a_idx][2] = ant_pos["Z"]

    def get_object_3dtraces(self):
        """
        ..warning:
           Memory duplication of traces, remove ZhairesSingleEventHdf5 object is necessary

        :param self:
        """
        self._get_traces()
        self._get_antspos()
        o_tevent = Handling3dTracesOfEvent(f"ZHAIRES simulation")
        #  MHz/ns: 1e-6/1e-9 = 1e3
        sampling_freq_mhz = 1e3 / self.t_sample_ns
        o_tevent.init_traces(
            self.traces,
            self.ants_id,
            self.t_start,
            sampling_freq_mhz,
        )
        o_tevent.init_network(self.ants_pos)
        o_tevent.set_unit_axis(r"$\mu$V/m", "cart")
        return o_tevent
