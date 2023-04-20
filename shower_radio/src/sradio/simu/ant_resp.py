"""
Created on 4 avr. 2023

@author: jcolley

Hypothesis: small network  (20-30km ) so => [N]~[DU] for vector/direction

"""


from logging import getLogger

import numpy as np
import sradio.basis.coord as coord



logger = getLogger(__name__)



class PreComputeInterpolLeff:
    """
    Precompute linear interpolation of frequency of Leff
    """

    def __init__(self):
        # index of freq in first in band 30-250MHz
        self.idx_first = None
        # index of freq in last plus one in band 30-250MHz
        self.idx_lastp1 = None
        # array of index where f_out are in f_in
        self.idx_itp = None
        # array of coefficient inf
        self.c_inf = None
        # array of coefficient sup
        # c_inf + c_sup = 1
        self.c_sup = None

    def init_linear_interpol(self, freq_in_mhz, freq_out_mhz):
        """
        Precompute coefficient of linear interpolation for freq_out_mhz with reference defined at freq_in_mhz

        :param freq_in_mhz: regular array of frequency where function is defined
        :param freq_out_mhz: regular array frequency where we want interpol
        """
        d_freq_out = freq_out_mhz[1]
        # index of freq in first in band, + 1 to have first in band
        idx_first = int(freq_in_mhz[0] / d_freq_out) + 1
        # index of freq in last plus one, + 1 to have first out band
        idx_lastp1 = int(freq_in_mhz[-1] / d_freq_out) + 1
        self.idx_first = idx_first
        self.idx_lastp1 = idx_lastp1
        d_freq_in = freq_in_mhz[1] - freq_in_mhz[0]
        freq_in_band = freq_out_mhz[idx_first:idx_lastp1]
        self.idx_itp = np.trunc((freq_in_band - freq_in_mhz[0]) / d_freq_in).astype(int)
        # define coefficient of linear interpolation
        self.c_sup = (freq_in_band - freq_in_mhz[self.idx_itp]) / d_freq_in
        self.c_inf = 1 - self.c_sup

    def get_linear_interpol(self, a_val):
        """
        Return f(freq_out_mhz) by linear interpolation of f defined by
        f(freq_in_mhz) = a_val

        :param a_val: defined value of function at freq_in_mhz
        """
        a_itp = self.c_inf * a_val[self.idx_itp] + self.c_sup * a_val[self.idx_itp + 1]
        return a_itp


class LengthEffProcessing:
    def __init__(self, name, leff_sampling, o_pre):
        self.name = name
        self.data = leff_sampling
        self.o_pre = o_pre

    def _get_idx_interpol_sph(self):
        # delta theta in degree
        data = self.data
        phi_efield = self.dir_src_deg[0]
        theta_efield = self.dir_src_deg[1]
        dtheta = data.theta_deg[1] - data.theta_deg[0]
        # theta_efield between index it0 and it1 in theta antenna response representation
        rt1 = (theta_efield - data.theta_deg[0]) / dtheta
        # prevent > 360 deg or >180 deg ?
        it0 = int(np.floor(rt1) % data.theta_deg.size)
        it1 = it0 + 1
        if it1 == data.theta_deg.size:  # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= np.floor(rt1)
        rt0 = 1 - rt1
        # phi_efield between index ip0 and ip1 in phi antenna response representation
        dphi = data.phi_deg[1] - data.phi_deg[0]  # deg
        rp1 = (phi_efield - data.phi_deg[0]) / dphi
        ip0 = int(np.floor(rp1) % data.phi_deg.size)
        ip1 = ip0 + 1
        if ip1 == data.phi_deg.size:  # Results are periodic along phi
            ip1 = 0
        rp1 -= np.floor(rp1)
        rp0 = 1 - rp1
        # weight = [rt0, rt1, rp0, rp1]
        # idx_i = [it0, it1, ip0, ip1]
        return rt0, rt1, rp0, rp1, it0, it1, ip0, ip1

    def set_dir_source(self, sph_du):
        self.dir_src_deg = np.rad2deg(sph_du)
        self.dir_src_rad = sph_du

    def set_angle_polar(self, a_pol):
        self.angle_pol = a_pol

    def get_fft_leff_du(self):
        l_p, l_t = self.get_fft_leff_tan()
        p_rad = self.dir_src_rad[0]
        t_rad = self.dir_src_rad[1]
        c_t, s_t = np.cos(t_rad), np.sin(t_rad)
        c_p, s_p = np.cos(p_rad), np.sin(p_rad)
        l_x = l_t * c_t * c_p - s_p * l_p
        l_y = l_t * c_t * s_p + c_p * l_p
        l_z = -s_t * l_t
        return l_x, l_y, l_z

    def get_fft_leff_tan(self):
        rt0, rt1, rp0, rp1, it0, it1, ip0, ip1 = self._get_idx_interpol_sph()
        leff = self.data.leff_theta
        leff_itp_t = (
            rp0 * rt0 * leff[ip0, it0, :]
            + rp1 * rt0 * leff[ip1, it0, :]
            + rp0 * rt1 * leff[ip0, it1, :]
            + rp1 * rt1 * leff[ip1, it1, :]
        )
        leff = self.data.leff_phi
        leff_itp_p = (
            rp0 * rt0 * leff[ip0, it0, :]
            + rp1 * rt0 * leff[ip1, it0, :]
            + rp0 * rt1 * leff[ip0, it1, :]
            + rp1 * rt1 * leff[ip1, it1, :]
        )
        leff_itp_sph = np.array([leff_itp_t, leff_itp_p])
        pre = self.o_pre
        leff_itp = (
            pre.c_inf * leff_itp_sph[:, pre.idx_itp] + pre.c_sup * leff_itp_sph[:, pre.idx_itp + 1]
        )
        # now add zeros outside leff frequency band and unpack leff theta , phi
        l_t = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
        l_t[pre.idx_first : pre.idx_lastp1] = leff_itp[0]
        l_p = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
        l_p[pre.idx_first : pre.idx_lastp1] = leff_itp[1]
        return l_p, l_t

    def get_fft_leff_pol(self):
        l_p, l_t = self.get_fft_leff_tan()
        return np.cos(self.angle_pol) * l_p + np.sin(self.angle_pol) * l_t


class DetectorUnitAntenna3Axis:
    """
    Compute DU response at efield



    """

    def __init__(self):
        """

        :param name:
        :param pos_n: [m] (3,) in stations frame [N]
        """
        self.name = "TBD"
        self.pos_du_n = np.zeros(3, dtype=np.float32)
        self.o_pre = PreComputeInterpolLeff()

    def set_name_pos(self, name, pos_n):
        """

        :param name:
        :param pos_n: [m] (3,) in stations frame [N]
        """
        self.name = name
        self.pos_du_n = pos_n

    def set_dict_leff(self, d_leff):
        """
        set object LengthEffProcessing
        """
        # all l_eff share frequency precompute o_pre
        self.l_leff = [1,2,3]
        axis = "sn"
        self.l_leff[0] = LengthEffProcessing(axis, d_leff[axis], self.o_pre)
        axis = "ew"
        self.l_leff[1] = LengthEffProcessing(axis, d_leff[axis], self.o_pre)
        axis = "up"
        self.l_leff[2] = LengthEffProcessing(axis, d_leff[axis], self.o_pre)

    def set_freq_out_mhz(self, a_freq):
        self.freq_out_mhz = a_freq
        freq_in_mhz = self.l_leff[0].data.freq_mhz
        self.o_pre.init_linear_interpol(freq_in_mhz, a_freq)

    def set_pos_source(self, pos_n):
        """
        set of source mainly Xmax in [N]

        :param pos_n:
        :type pos_n:
        """
        self.pos_src_n = pos_n

    def update_dir_source(self):
        """
        return direction of source in [DU] frame
        :param self:
        :type self:
        """
        diff_n = self.pos_src_n - self.pos_du_n
        # Hypothesis: small network  (20-30km ) => [N]~[DU] for vector/direction             
        self.dir_src_du = coord.frame_du_cart_to_sph(diff_n)
        logger.debug(f"phi, d_zen = {np.rad2deg(self.dir_src_du)}")

    def get_resp_3d_efield_du(self, efield_du):
        """Return fft of antennas response for 3 axis with efield in [N] frame

        :param efield_du: electric field at DU in [N]/[DU]
        :type efield_du: (3, n_s)::float32
        """
        pass

    def get_resp_2d_efield_tan(self, efield_tan):
        """Return fft of antennas response for 3 axis with efield in [TAN] tangential plan

        :param efield_tan: electric field in [TAN]
        :type efield_tan: (2, n_s)::float32
        """
        pass

    def get_resp_1d_efield_pol(self, efield_pol):
        """Return fft of antennas response for 3 axis with efield in [POL] linear polarization

        :param efield_pol:electric field in [POL]
        :type efield_pol: (n_s,)::float32
        """
        pass
