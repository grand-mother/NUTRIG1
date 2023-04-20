"""
Simulation of galaxy emission in radio frequency through antennas GP300 in voltage

From github grand-mother/grand 

..Authors:
    PengFei and Xidian group GRAND collaboration
"""

import h5py
import numpy as np

from sradio.num.signal import interpol_at_new_x


class GalaxySignalThroughGp300:
    """
    Return galaxy signal directly in voltage as GP300 antenna see it
    """

    def __init__(self, path_model):
        gala_show = h5py.File(path_model, "r")
        self.gala_voltage = np.transpose(
            gala_show["v_amplitude"]
        )  # SL, microV per MHz, seems to be Vmax=sqrt(2*mean(V*V)), not std(V)=sqrt(mean(V*V))
        # gala_power_mag = np.transpose(gala_show["p_narrow"])
        self.gala_freq = gala_show["freq_all"]

    def get_volt_all_du(self, f_lst, size_out, freqs_mhz, nb_ant, seed=None):
        """Return for all DU fft of galaxy signal in voltage
        
        This program is used as a subroutine to complete the calculation and
        expansion of galactic noise

        ..Authors:
          PengFei and Xidian group

        :param f_lst: select the galactic noise LST at the LST moment
        :type f_lst: float
        :param size_out: is the extended length
        :type size_out:int
        :param freqs_mhz: array of output frequencies
        :type freqs_mhz:float (nb freq,)
        :param nb_ant: number of antennas
        :type nb_ant:int
        :param show_flag: print figure
        :type show_flag: boll
        :param seed: if None, values are randomly generated as expected.
        :type seed:if number, same set of randomly generated output. This is useful for testing.
        
        :return: FFT of galactic noise for all DU and components
        :rtype: float(nb du, 3, nb freq)
        """
        # TODO: why lst is an integer ?
        lst = int(f_lst)
        v_amplitude_infile = self.gala_voltage[:, :, lst - 1]
        # SL
        nb_freq = len(freqs_mhz)
        freq_res = freqs_mhz[1] - freqs_mhz[0]
        v_amplitude_infile = v_amplitude_infile * np.sqrt(freq_res)
        v_amplitude = np.zeros((nb_freq, 3))
        v_amplitude[:, 0] = interpol_at_new_x(
            self.gala_freq[:, 0], v_amplitude_infile[:, 0], freqs_mhz
        )
        v_amplitude[:, 1] = interpol_at_new_x(
            self.gala_freq[:, 0], v_amplitude_infile[:, 1], freqs_mhz
        )
        v_amplitude[:, 2] = interpol_at_new_x(
            self.gala_freq[:, 0], v_amplitude_infile[:, 2], freqs_mhz
        )
        # RK: above loop is replaced by lines below. Also np.random.default_rng(seed) is used instead of np.random.seed().
        #     if seed is a fixed number, same set of randomly generated number is produced. This is useful for testing.
        v_amplitude = v_amplitude.T
        rng = np.random.default_rng(seed)
        amp = rng.normal(loc=0, scale=v_amplitude[np.newaxis, ...], size=(nb_ant, 3, nb_freq))
        phase = 2 * np.pi * rng.random(size=(nb_ant, 3, nb_freq))
        v_complex = np.abs(amp * size_out / 2) * np.exp(1j * phase)
        return v_complex
