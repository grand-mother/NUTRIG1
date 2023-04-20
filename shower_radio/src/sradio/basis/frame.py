"""

To convert between frame coordinate system is always cartesian

See coord.py module for specific spherical convention associated to each frame


Frame available:

    [N] is the frame of network stations, for small network [N]~[DU] for vector/direction


    [DU] is the frame associated to a DU :
        * Cartesian
          * X: North mag, Y: West mag, Z: Up
        * Spherical
          * azi_w (phi_du) = angle between X and azi_w(West)=90 degree
          * d_zen (theta_du) = angle from zenith , d_zen(horizon)=90 degree

    [TAN] is the frame associated to a DU tangential at source direction
          * e_phi ?? , e_theta???, e_up


    Notation:
       convention xxx_yy variable means position of xxx is in [yy] frame.
       exemple : efield_tan is E field in tangential frame of antenna

"""


import numpy as np


class FrameAFrameB:
    def __init__(self, offset_ab_a, rot_b2a):
        self.offset_ab_a = offset_ab_a
        self.rot_b2a = rot_b2a
        self.offset_ab_b = np.matmul(self.rot_b2a.T, offset_ab_a)

    def pos_to_a(self, pos_b):
        return self.offset_ab_a + np.matmul(self.rot_b2a, pos_b)

    def pos_to_b(self, pos_a):
        return -self.offset_ab_b + np.matmul(self.rot_b2a.T, pos_a)

    def vec_to_a(self, vec_b):
        return np.matmul(self.rot_b2a, vec_b)

    def vec_to_b(self, vec_a):
        return np.matmul(self.rot_b2a.T, vec_a)


class FrameDuFrameTangent(FrameAFrameB):
    def __init__(self, vec_dir_du):
        offset_ab_a = np.zeros(3, dtype=vec_dir_du.dtype)
        rot_b2a = 2
        super().__init__(offset_ab_a, rot_b2a)
