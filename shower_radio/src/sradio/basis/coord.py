"""
Coordinate transformation in same frame.

see frame.py module to have frame definition and specific convention of axis and angle
"""


import numpy as np


def frame_du_cart_to_sph(xyz_n):
    """Return [DU] spherical angle in rad from cartesian
            x to y  : azi_w = + 90
            z to x  : d_zen = + 90

    :param xyz_n:
    :type xyz_n:
    """
    # TODO: rewrite for vector input like (n,3)
    azi_w = np.arctan2(xyz_n[1], xyz_n[0])
    rho = np.sqrt(xyz_n[0] ** 2 + xyz_n[1] ** 2)
    d_zen = np.arctan2(rho, xyz_n[2])
    return np.array([azi_w, d_zen])


def frame_du_cart_to_sph_dist(xyz_n):
    """Return [DU] spherical angle in rad from cartesian
            x to y  : azi_w = + 90
            z to x  : d_zen = + 90

    :param xyz_n:
    :type xyz_n:
    """
    # TODO: rewrite for vector input like (n,3)
    azi_w = np.arctan2(xyz_n[1], xyz_n[0])
    rho_2 = xyz_n[0] ** 2 + xyz_n[1] ** 2
    rho = np.sqrt(rho_2)
    d_zen = np.arctan2(rho, xyz_n[2])
    return np.array([azi_w, d_zen, np.sqrt(rho_2 + xyz_n[2] ** 2)])


def frame_du_sph_to_cart(sph3):
    """Return [DU]"""
    pass
