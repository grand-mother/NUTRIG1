"""
Created on 4 avr. 2023

@author: jcolley
"""


from logging import getLogger

import numpy as np
import scipy.fft as sf

from sradio.basis.traces_event import Handling3dTracesOfEvent
from sradio.model.ant_resp import DetectorUnitAntenna3Axis, get_leff_from_files
from sradio.num.signal import get_fastest_size_rfft
from sradio.model.galaxy import GalaxySignalThroughGp300


logger = getLogger(__name__)


class SimuDetectorUnitResponse:
    """
    Simulation of
      * antenna response
      * electronic
      * galactic signal
    with E field input

    Processing to do:

      * Convolution in time domain : (Efield*(l_eff + noise))*IR_rf_chain

        * '*' is convolution operator
        * l_eff : effective length of antenna, ie impulsional response of antenna
        * noise: galactic noise at local sideral time
        * IR_rf_chain :  impulsional response of electronic chain

    Processing performed:

      * Calculation in Fourier space: (F_Efield.(L_eff + F_noise)).TF_rf_chain

        * in Fourier domain convolution becomes multiplication
        * '.' multiplication term by term
        * L_eff : effective length of antenna in Fourier space, ie transfer function
        * F_noise: FFT of galactic noise at local sideral time
        * TF_rf_chain : transfer function of electronic chain

      * We used a common frequency definition for all calculation stored in freqs_mhz attribute
        and computed with function get_fastest_size_rfft()

    .. note::
       * no IO, only memory processing
       * manage only one event
    """

    def __init__(self, path_leff, path_gal=""):
        """
        Constructor
        """
        # Parameters
        self.params = {"flag_add_gal": True, "flag_add_rf": False, "lst": 18.0}
        # object contents Efield and network information
        self.o_efield = Handling3dTracesOfEvent()
        self.rf_chain = None
        self.o_ant3d = DetectorUnitAntenna3Axis(get_leff_from_files(path_leff))
        if path_gal != "":
            self.o_gal = GalaxySignalThroughGp300(path_gal)
        else:
            self.params["flag_add_gal"] = False
            logger.info("No galaxy signal added")
        # object of class ShowerEvent
        self.o_shower = None
        # FFT info
        self.sig_size = 0
        self.fact_padding = 6
        #  size_with_pad ~ sig_size*fact_padding
        self.size_with_pad = 0
        # float (size_with_pad,) array of frequencies in MHz in Fourier domain
        self.freqs_out_mhz = 0
        # outputs
        self.fft_noise_gal_3d = None
        self.v_out = None

    ### SETTER

    def set_data_efield(self, tr_evt):
        """

        :param tr_evt: object contents Efield and network information
        :type tr_evt: Handling3dTracesOfEvent
        """
        assert isinstance(tr_evt, Handling3dTracesOfEvent)
        self.o_efield = tr_evt
        self.v_out = np.zeros_like(self.o_efield.traces)
        logger.debug(self.v_out.shape)
        self.sig_size = self.o_efield.get_size_trace()
        # common frequencies for all processing in Fourier domain
        self.size_with_pad, self.freqs_out_mhz = get_fastest_size_rfft(
            self.sig_size,
            self.o_efield.f_samp_mhz,
            self.fact_padding,
        )
        logger.debug(self.size_with_pad)
        logger.debug(self.sig_size)
        # precompute interpolation for all antennas
        logger.info("Precompute weight for linear interpolation of Leff in frequency")
        self.o_ant3d.set_freq_out_mhz(self.freqs_out_mhz)
        self.fft_efield = sf.rfft(self.o_efield.traces, n=self.size_with_pad)
        assert self.fft_efield.shape[0] == self.o_efield.traces.shape[0]
        assert self.fft_efield.shape[1] == self.o_efield.traces.shape[1]
        # compute total transfer function of RF chain
        # self.rf_chain.compute_for_freqs(self.freqs_out_mhz)
        if self.params["flag_add_gal"]:
            # lst: local sideral time, galactic noise max at 18h
            logger.info("Compute galaxy noise for all traces")
            self.fft_noise_gal_3d = self.o_gal.get_volt_all_du(
                self.params["lst"],
                self.size_with_pad,
                self.freqs_out_mhz,
                self.o_efield.get_nb_du(),
            )
    
    def set_xmax(self, xmax_xc):
        """
        
        :param xmax_xc: position Xmax  in frame [XCore]  
        :type xmax_xc: float (3,)
        """
        self.o_ant3d.set_pos_source(xmax_xc)

    def set_data_shower(self, shower):
        """

        :param shower: object contents shower parameters
        :type shower: ShowerEvent
        """
        self.shower = shower
        self.o_ant3d.set_pos_source(shower["xmax"])

    ### GETTER / COMPUTER

    def compute_du_all(self):
        """
        Simulate all DU
        """
        for idx in range(self.o_efield.get_nb_du()):
            self.compute_du_idx(idx)

    def compute_du_idx(self, idx_du):
        """Simulate one DU
        Simulation DU effect computing for DU at idx

        Processing order:

          1. antenna responses
          2. add galactic noise
          3. RF chain effect

        :param idx_du: index of DU in array traces
        :type  idx_du: int
        """
        logger.info(f"==============>  Processing DU with id: {self.o_efield.du_id[idx_du]}")
        self.o_ant3d.set_name_pos(self.o_efield.du_id[idx_du], self.o_efield.network.du_pos[idx_du])
        ########################
        # 1) Antenna responses
        ########################
        # Voltage open circuit
        fft_3d = self.o_ant3d.get_resp_3d_efield_du(self.fft_efield[idx_du])
        ########################
        # 2) Add galactic noise
        ########################
        if self.params["flag_add_gal"]:
            # noise_gal = sf.irfft(self.fft_noise_gal_3d[idx_du])[:, : self.sig_size]
            # logger.debug(np.std(noise_gal, axis=1))
            # self.voc[idx_du] += noise_gal
            fft_3d += self.fft_noise_gal_3d[idx_du]
            raise
        ########################
        # 3) RF chain
        ########################
        if self.params["flag_add_rf"]:
            fft_3d *= self.rf_chain.get_tf_3d()
            raise
        # inverse FFT and remove zero-padding
        # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
        self.v_out[idx_du] = sf.irfft(fft_3d)[:, : self.sig_size]
