"""

To convert between frame coordinate system is always cartesian

See coord.py module for specific spherical convention associated to each frame


Frame available:
================

    [W84] for WGS84 is the frame of GPS
        * Remark : use to define position of DU, 
                   for shower physic with time measure, position of DU arround 10 cm seems ok
        * Origin : center of earth
        * Cartesian: ECEF
          * X: on equator longitude 0, Y: on equator to East, Z: ~rotation axis
        * Spherical with GRS80 ellipsoide
          * longitude geodetic
          * latitide geodetic
          * altitude:
              * above ellipsoide or geoide EGM96
    

    [N] is the frame associated to network stations
        * Origin [W84]: can be center of network or another position like xcore
        * Cartesian: NWU ie tangential to the surface of the earth
          * X: North mag, Y: West mag, Z: normal up to earth 
        * Spherical
          * azi_w (phi_n)[0,360] = angle between X and azi_w(West)=90 degree
          * d_zen (theta_n) = angle from zenith , d_zen(horizon)=90 degree
        * Remark : so in fact it's a familly of frame 
        * Example: ZHaireS simulation 
        

    [DU] is the frame associated to one DU 
        * Origin [W84]/[N]: antenna position given by GPS position
        * Remark : normaly we must indicate the id of the DU, like [DUi] but as we don't have 
                   computation between DU it's not necessary to specify it
        * Cartesian: NWU ie tangential to the surface of the earth
          * X: North mag, Y: West mag, Z: Up
        * Spherical
          * azi_w (phi_du) [0,360] = angle between X and azi_w(West)=90 degree
          * d_zen (theta_du) = angle from zenith , d_zen(horizon)=90 degree


    [TAN] is the frame associated to a DU tangential at 
          source direction (phi_src, theta_src) 
        * Origin [DU]: position associated with unit vector with direction (phi_src, theta_src)
        * Cartesian: tangential to the unit sphere around antenna
          * X: e_phi, Y: e_theta, Z: normal up to sphere
        * Spherical
          * None


    Remark:
       In case of small network (20-30km) [N] and [DU] are equivalent for vector orientation 
       because local normal and magnetic field can be considered as constant on this aera.


    Notation:
       convention xxx_yy variable means position of xxx is in [yy] frame.
       example : efield_tan is E field in tangential frame of antenna

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
