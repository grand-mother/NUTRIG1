"""
Coordinate transformation in same frame.

see frame.py module to have frame definition and specific convention of axis and angle
"""


import numpy as np


#
# [DU]
#


def cart_to_dir_du(xyz):
    """Convert cartesian vector xyz to direction in [DU] frame

    :param xyz: cartesian vector
    :type xyz: float (3,)

    :return: angle direction: azimuth, distance zenithal
    :rtype: float (2,)
    """
    azi_w = np.arctan2(xyz[1], xyz[0])
    rho = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
    d_zen = np.arctan2(rho, xyz[2])
    return np.array([azi_w, d_zen])


def cart_to_sph_du(xyz):
    """Convert cartesian vector xyz to spherical in [DU] frame

    :param xyz: cartesian vector
    :type xyz: float (3,)

    :return: azimuth, distance zenithal, norm
    :rtype: float (3,)
    """
    # TODO: rewrite for vector input like (n,3)
    azi_w = np.arctan2(xyz[1], xyz[0])
    rho_2 = xyz[0] ** 2 + xyz[1] ** 2
    rho = np.sqrt(rho_2)
    d_zen = np.arctan2(rho, xyz[2])
    return np.array([azi_w, d_zen, np.sqrt(rho_2 + xyz[2] ** 2)])


def sph_to_cart_du(sph):
    """Convert spherical vector xyz to cartesian in [DU] frame

    :param sph: azimuth, distance zenithal, norm
    :type sph: float (3,)

    :return: cartesian vector
    :rtype: float (3,)
    """
    c_a, s_a = np.cos(sph[0]), np.sin(sph[0])
    c_dz, s_dz = np.cos(sph[1]), np.sin(sph[1])
    xyz = np.empty_like(sph)
    xyz[0] = s_dz * c_a
    xyz[1] = s_dz * s_a
    xyz[2] = c_dz
    return sph[2] * xyz
