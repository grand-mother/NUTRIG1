"""
Handling a set of 3D traces
"""
from logging import getLogger

import numpy as np
import matplotlib.pylab as plt

from .traces_event import Handling3dTracesOfEvent


logger = getLogger(__name__)


def efield_unit_polar_vec(trace, threasold=20):
    """

    :param trace:
    :type trace: float (n_s,3)
    :param threasold:
    :type threasold:
    """
    # TODO: too must .T , find something more simple ...
    assert trace.shape[1] == 3
    n_elec = np.linalg.norm(trace, axis=1)
    idx_hb = np.where(n_elec > threasold)[0]
    # unit
    sple_ok = trace[idx_hb].T / n_elec[idx_hb]
    idx_neg = np.where(sple_ok[1] < 0)[0]
    sple_ok = sple_ok.T
    sple_ok[idx_neg] = -sple_ok[idx_neg]
    n_elec_2 = n_elec[idx_hb] * n_elec[idx_hb]
    temp = sple_ok.T * n_elec_2
    logger.info(temp.shape)
    pol_est = np.sum(temp, axis=1) / np.sum(n_elec_2)
    pol_est /= np.linalg.norm(pol_est)
    logger.info(pol_est.shape)
    logger.info(pol_est)
    logger.info(np.linalg.norm(pol_est))
    # TODO: check if lineat polarization is here with stats test
    return pol_est


def efield_3d_to_1dpolar(efield3d, threasold=20):
    """

    :param efield3d: [uV/m] Efield 3D
    :type efield3d: float (3, n_s)
    :param threasold: [uV/m] used to select sample to fit direction 
    :type threasold: float
    """
    pol_est = efield_unit_polar_vec(efield3d.T, threasold)
    efield1d = np.dot(efield3d.T, pol_est)
    # plt.figure()
    # plt.plot(efield1d)
    return efield1d, pol_est


class HandlingEfieldOfEvent(Handling3dTracesOfEvent):
    """
    Handling a set of E field traces associated to one event observed on Detector Unit network

    Goal: apply specific E field processing on all traces
    """

    def __init__(self, name="NotDefined"):
        super().__init__("E_fied " + name)

    ### INTERNAL

    ### INIT/SETTER
