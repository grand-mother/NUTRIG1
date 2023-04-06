"""
Estimate Xmax position with plane wavefront model (pwf)
"""
from logging import getLogger

import numpy as np
import scipy.optimize as so

from sradio.num.const import C_LIGHT_MS

logger = getLogger(__name__)


def solve_with_plane_model(pos_ant, time_max):
    """

    :param params_in:
    :type params_in:
    :param pos_ant: [m]
    :type pos_ant: float (nb_ant,3)
    :param time_max: [s]
    :type time_max: float (nb_ant,)
    """
    args = (pos_ant, time_max * C_LIGHT_MS)
    params_in = np.array([3 * np.pi / 4.0, np.pi])
    res = so.minimize(
        pwf_loss, params_in, jac=pwf_grad, hess=pwf_hess, args=args, method="Newton-CG"
    )
    params_out = res.x
    chi2 = pwf_loss(params_out, *args)
    return params_out, chi2, res


def pwf_loss(params, Xants, tants, cr=1.0, verbose=False):
    """
    Defines Chi2 by summing model residuals
    over antenna pairs (i,j):
    loss = \sum_{i>j} ((Xants[i,:]-Xants[j,:]).K - cr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants,3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants,))
    cr is radiation speed, by default 1 since time is expressed in m.
    """

    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st * cp, st * sp, ct])
    # Make sure tants and Xants are compatible
    if Xants.shape[0] != nants:
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Use numpy outer methods to build matrix X_ij = x_i -x_j
    xk = np.dot(Xants, K)
    DXK = np.subtract.outer(xk, xk)
    DT = np.subtract.outer(tants, tants)
    chi2 = (
        (DXK - cr * DT) ** 2
    ).sum() / 2.0  # Sum over upper triangle, diagonal is zero because of antisymmetry of DXK, DT
    if verbose:
        print("Chi2 = ", chi2)
    return chi2


def pwf_grad(params, Xants, tants, cr=1.0, verbose=False):

    """
    Gradient of pwf_loss, with respect to theta, phi
    """
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st * cp, st * sp, ct])

    xk = np.dot(Xants, K)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK = np.subtract.outer(xk, xk)
    DT = np.subtract.outer(tants, tants)
    RHS = DXK - cr * DT

    # Derivatives of K w.r.t. theta, phi
    dKdtheta = np.array([ct * cp, ct * sp, -st])
    dKdphi = np.array([-st * sp, st * cp, 0.0])
    xk_theta = np.dot(Xants, dKdtheta)
    xk_phi = np.dot(Xants, dKdphi)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK_THETA = np.subtract.outer(xk_theta, xk_theta)
    DXK_PHI = np.subtract.outer(xk_phi, xk_phi)

    jac_theta = np.sum(
        DXK_THETA * RHS
    )  # Factor of 2 of derivatives compensates ratio of sum to upper diag sum
    jac_phi = np.sum(DXK_PHI * RHS)
    if verbose:
        print("Jacobian = ", jac_theta, jac_phi)
    return np.array([jac_theta, jac_phi])


def pwf_hess(params, Xants, tants, cr=1.0, verbose=False):
    """
    Hessian of pwf_loss, with respect to theta, phi
    """
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st * cp, st * sp, ct])

    xk = np.dot(Xants, K)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK = np.subtract.outer(xk, xk)
    DT = np.subtract.outer(tants, tants)
    RHS = DXK - cr * DT

    # Derivatives of K w.r.t. theta, phi
    dK_dtheta = np.array([ct * cp, ct * sp, -st])
    dK_dphi = np.array([-st * sp, st * cp, 0.0])
    d2K_dtheta = np.array([-st * cp, -st * sp, -ct])
    d2K_dphi = np.array([-st * cp, -st * sp, 0.0])
    d2K_dtheta_dphi = np.array([-ct * sp, ct * cp, 0.0])

    xk_theta = np.dot(Xants, dK_dtheta)
    xk_phi = np.dot(Xants, dK_dphi)
    xk2_theta = np.dot(Xants, d2K_dtheta)
    xk2_phi = np.dot(Xants, d2K_dphi)
    xk2_theta_phi = np.dot(Xants, d2K_dtheta_dphi)

    # Use numpy outer method to buid matrix X_ij = x_i - x_j
    DXK_THETA = np.subtract.outer(xk_theta, xk_theta)
    DXK_PHI = np.subtract.outer(xk_phi, xk_phi)
    DXK2_THETA = np.subtract.outer(xk2_theta, xk2_theta)
    DXK2_PHI = np.subtract.outer(xk2_phi, xk2_phi)
    DXK2_THETA_PHI = np.subtract.outer(xk2_theta_phi, xk2_theta_phi)

    hess_theta2 = np.sum(DXK2_THETA * RHS + DXK_THETA ** 2)
    hess_phi2 = np.sum(DXK2_PHI * RHS + DXK_PHI ** 2)
    hess_theta_phi = np.sum(DXK2_THETA_PHI * RHS + DXK_THETA * DXK_PHI)

    return np.array([[hess_theta2, hess_theta_phi], [hess_theta_phi, hess_phi2]])
