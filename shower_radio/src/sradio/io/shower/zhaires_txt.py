"""
Created on 28 mars 2023

@author: jcolley


Read ZHAires Outputs simulation, can be convert:
* object 3D traces
* ASDF file

"""

import re
import os
import os.path
from logging import getLogger

import numpy as np
import asdf

from sradio.basis.traces_event import Handling3dTracesOfEvent

logger = getLogger(__name__)

# approximative regular expression of string float
REAL = r"[+-]?[0-9][0-9.eE+-]*"


def convert_str2number(elmt):
    """
    Try convert string value of dictionary in float with recursive scheme

    :param elmt:
    """
    if isinstance(elmt, str):
        try:
            if "." in elmt or "e" in elmt or "E" in elmt:
                return float(elmt)
            else:
                return int(elmt)
        except ValueError:
            return elmt
    elif isinstance(elmt, dict):
        return {key: convert_str2number(val) for key, val in elmt.items()}
    elif isinstance(elmt, list):
        return [convert_str2number(val) for val in elmt]
    else:
        return elmt


class ZhairesSummaryFileVers28:
    def __init__(self, file_sry="", str_sry=""):
        self.d_sry = {}
        self.l_error = []
        self.d_re = {
            "vers_aires": r"This is AIRES version\s+(?P<vers_aires>\w+\.\w+\.\w+)\s+\(",
            "vers_zhaires": r"With ZHAireS version (?P<vers_zhaires>\w+\.\w+\.\w+) \(",
            "primary": r"Primary particle:\s+(?P<primary>\w+)\s+",
            "site": fr"Site:\s+(?P<name>\w+)\s+\(Lat:\s+(?P<lat>{REAL})\s+deg. Long:\s+(?P<lon>{REAL})\s+deg",
            "geo_mag": fr"Geomagnetic field: Intensity:\s+(?P<norm>{REAL})\s+(?P<unit>\w+)\s+I:\s+(?P<inc>{REAL})\s+deg. D:\s+(?P<dec>{REAL})\s+deg",
            "energy": fr"Primary energy:\s+(?P<value>{REAL})\s+(?P<unit>\w+)",
            "zenith_angle": fr"Primary zenith angle:\s+(?P<zenith_angle>{REAL})\s+deg",
            "azimuth_angle": fr"Primary azimuth angle:\s+(?P<azimuth_angle>{REAL})\s+deg",
            "x_max": fr"Location of max\.\((?P<unit>\w+)\):\s+{REAL}\s+{REAL}\s+(?P<x>{REAL})\s+(?P<y>{REAL})\s+(?P<z>{REAL})\s+",
            "t_sample_ns": fr"Time bin size:\s+(?P<t_sample_ns>{REAL})ns",
        }
        self.str_sry = str_sry
        if file_sry != "":
            with open(file_sry) as f_sry:
                self.str_sry = f_sry.read()

    def extract_all(self):
        self.l_error = []
        d_sry = {}
        for key, s_re in self.d_re.items():
            ret = re.search(s_re, self.str_sry)
            logger.debug(ret)
            if ret:
                d_ret = ret.groupdict()
                if key in d_ret.keys():
                    # single value
                    d_sry.update(d_ret)
                else:
                    # set of values in sub dictionary with key {key}
                    d_sry[key] = d_ret
            else:
                logger.error(
                    f"Can't find '{key}' information with this regular expression:\n{s_re}"
                )
                self.l_error.append(key)
        self.d_sry = convert_str2number(d_sry)

    def get_dict(self):
        return self.d_sry

    def is_ok(self):
        return len(self.l_error) == 0


class ZhairesSummaryFileVers28b(ZhairesSummaryFileVers28):
    def __init__(self, file_sry="", str_sry=""):
        super().__init__(file_sry, str_sry)
        self.d_re[
            "x_max"
        ] = fr"Pos. Max.:\s+{REAL}\s+{REAL}\s+(?P<x>{REAL})\s+(?P<y>{REAL})\s+(?P<z>{REAL})\s+"


# add here all version of ZHaireS summary file
L_SRY_VERS = [ZhairesSummaryFileVers28b, ZhairesSummaryFileVers28]


class ZhairesSingleEventBase:
    def get_dict(self):
        d_gen = self.d_info.copy()
        d_gen["traces"] = self.traces
        d_gen["t_start"] = self.t_start
        d_gen["ant_pos"] = self.ants
        return d_gen

    def write_asdf_file(self, p_file):
        df_simu = asdf.AsdfFile(self.get_dict())
        df_simu.write_to(p_file, all_array_compression="zlib")


class ZhairesSingleEventText(ZhairesSingleEventBase):
    def __init__(self, path_zhaires):
        self.path = path_zhaires
        self.read_summary_file()
        self.read_antpos_file()
        self.read_trace_files()

    def add_path(self, file):
        return os.path.join(self.path, file)

    def read_antpos_file(self):
        a_dtype = {
            "names": ("idx", "name", "x", "y", "z"),
            "formats": ("i4", "S20", "f4", "f4", "f4"),
        }
        self.ants = np.loadtxt(self.add_path("antpos.dat"), dtype=a_dtype)
        self.nb_ant = self.ants.shape[0]

    def read_summary_file(self):
        # l_files = list(filter(os.path.isfile, os.listdir(self.path)))
        l_files = os.listdir(self.path)
        # print(l_files)
        l_sry = []
        for m_file in l_files:
            if ".sry" in m_file:
                l_sry.append(m_file)
        nb_sry = len(l_sry)
        if nb_sry > 1:
            logger.warning(f"several files summary ! in {self.path}")
            logger.warning(l_sry)
        if nb_sry == 0:
            logger.error(f"no files summary ! in {self.path}")
            raise
        else:
            f_sry = self.add_path(l_sry[0])
            for sry_vers in L_SRY_VERS:
                sry = sry_vers(f_sry)
                sry.extract_all()
                if sry.is_ok():
                    self.d_info = sry.get_dict()
                    return
            logger.error("Unknown summary file version")
            raise

    def read_trace_files(self):
        trace_0 = np.loadtxt(self.add_path("a0.trace"))
        nb_sample = trace_0.shape[0]
        self.traces = np.empty((self.nb_ant, 3, nb_sample), dtype=np.float32)
        self.t_start = np.empty(self.nb_ant, dtype=np.float64)
        for idx in range(self.nb_ant):
            f_trace = self.add_path(f"a{idx}.trace")
            # print(f_trace)
            trace = np.loadtxt(f_trace)
            assert trace.shape[0] == nb_sample
            self.traces[idx] = trace.transpose()[1:]
            self.t_start[idx] = trace[0, 0]

    def get_object_3dtraces(self):
        o_tevent = Handling3dTracesOfEvent(f"ZHAIRES simulation")
        du_id = range(self.nb_ant)
        #  MHz/ns: 1e-6/1e-9 = 1e3
        sampling_freq_mhz = 1e3 / self.d_info["t_sample_ns"]
        o_tevent.init_traces(
            self.traces,
            du_id,
            self.t_start,
            sampling_freq_mhz,
        )
        ants = np.empty((self.nb_ant, 3), dtype=np.float32)
        ants[:, 0] = self.ants["x"]
        ants[:, 1] = self.ants["y"]
        ants[:, 2] = self.ants["z"]
        o_tevent.init_network(self.ants)
        o_tevent.set_unit_axis(r"$\mu$V/m", "cart")
        return o_tevent
