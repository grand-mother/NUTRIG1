"""
Estimate Xmax position with spherical wavefront model (swf)
"""
import numpy as np
import scipy.optimize as so

from numba import njit

from sradio.num.const import C_LIGHT_MS
from sradio.recons.shower_plane import solve_with_plane_model
from sradio.basis.du_network import DetectorUnitNetwork

R_earth = 6371007.0
ns = 325
kr = -0.1218
inv_kr = 1 / kr
# TODO: to replace by antenna position information, can be different for each antenna ...
# groundAltitude = 1086.0
groundAltitude = 2034

kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}


class ReconsXmaxSphericalModel:
    """
    All position defined in antenna frame
    """

    def __init__(self, pos_ant, time_max_s):
        self.pos_ant = pos_ant
        self.time_max = time_max_s
        self.rep_solver = so.OptimizeResult()
        self.net = DetectorUnitNetwork("Shower reconstruction")
        self.net.init_pos_id(pos_ant)

    def solve_xmax(self):
        params_out, chi2, res = solve_with_spheric_model(self.pos_ant, self.time_max)
        self.params_out = params_out
        self.chi2 = chi2
        self.rep_solver = res

    def get_status_solver(self):
        status = f"chi2= {self.chi2}"
        status += f"\nStatus solver= {self.rep_solver.success}"
        return status

    def get_wave_front_dir_propagation(self):
        """
        unit vector opposite of Xmax vector

        :param self:
        :type self:
        """
        pass

    def get_xmax(self):
        theta, phi, r_xmax, t_s = self.params_out
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        Xmax = -r_xmax * np.array([st * cp, st * sp, ct])
        mean_alt = self.pos_ant[:, 2].mean()
        print("mean_alt: ", mean_alt)
        Xmax += np.array([0.0, 0.0, mean_alt])
        return Xmax

    def plot_time_max(self):
        self.net.plot_footprint_1d(self.time_max, "Time max", scale="lin")

    def plot_residu(self):
        pass

    def plot_confidence_interval(self):
        pass


def solve_with_spheric_model(pos_ant, time_max, guess=None):
    """

    :param pos_ant:
    :type pos_ant: float (nb du, 3) [m]
    :param time_max:
    :type time_max: float [s]
    :param guess: direction of progression of shower ie
                  theta, phi ( where theta zenithal distance)
    :type guess: float (2,) [rad]
    """
    # Guess parameters
    if guess is None:
        dir_shower, chi2, res = solve_with_plane_model(pos_ant, time_max)
        # TODO: check return ?
        assert res.success
        print(res)
        dir_shower = np.rad2deg(dir_shower)
        theta_in = dir_shower[0]
        phi_in = dir_shower[1]
    else:
        theta_in = guess[0]
        phi_in = guess[1]
    # TODO: a voir avec SP.
    # theta, phi, r_xmax, t_s = params
    bounds = [
        [np.deg2rad(theta_in - 1), np.deg2rad(theta_in + 1)],
        [np.deg2rad(phi_in - 1), np.deg2rad(phi_in + 1)],
        [
            -15.6e3 - 12.3e3 / np.cos(np.deg2rad(theta_in)),
            -6.1e3 - 15.4e3 / np.cos(np.deg2rad(theta_in)),
        ],
        [6.1e3 + 15.4e3 / np.cos(np.deg2rad(theta_in)), 0],
    ]
    # TODO: angle and modulo ?
    params_in = np.array(bounds).mean(axis=1)
    params_in = np.zeros(4, dtype=np.float64)
    params_in[:2] = np.deg2rad([theta_in, phi_in])
    params_in[2] = 5000.0
    params_in[3] = np.min(time_max) - params_in[2] / C_LIGHT_MS
    print("params_in  angle= ", np.rad2deg(params_in[:2]))
    print("params_in others= ", params_in[2:])
    # print("pos ant:\n", pos_ant)
    # print("t evt  :\n", time_max)
    args = (pos_ant, time_max * C_LIGHT_MS)
    # , hess=swf_hess
    # jac=swf_grad,
    res = so.minimize(swf_loss, params_in, args=args, method="BFGS")
    # TODO: check return ?
    params_out = res.x
    chi2 = swf_loss(params_out, *args)
    return params_out, chi2, res


@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0, Xa):
    R02 = X0[0] * X0[0] + X0[1] * X0[0]
    # Altitude of emission in km
    temp = X0[2] + R_earth
    h0 = (np.sqrt(temp * temp + R02) - R_earth) / 1e3
    # print(h0)
    # Refractivity at emission
    rh0 = ns * np.exp(kr * h0)
    # modr = np.np.sqrt(R02)
    # print(modr)
    # if (modr > 1e3):
    if R02 > 1e6:
        modr = np.sqrt(R02)
        # Vector between antenna and emission point
        U = Xa - X0
        # Divide into pieces shorter than 10km
        nint = np.int(modr / 2e4) + 1
        K = U / nint
        # Current point coordinates and altitude
        Curr = X0
        currh = h0
        s = 0.0
        for i in range(nint):
            Next = Curr + K  # Next point
            nextR2 = Next[0] * Next[0] + Next[1] * Next[1]
            temp2 = Next[2] + R_earth
            nexth = (np.sqrt(temp2 * temp2 + nextR2) - R_earth) / 1e3
            diff = nexth - currh
            if np.abs(diff) > 1e-10:
                s += inv_kr * (np.exp(kr * nexth) - np.exp(kr * currh)) / diff
            else:
                s += np.exp(kr * currh)
            Curr = Next
            currh = nexth
            # print (currh)
        avn = ns * s / nint
        # print(avn)
        n_eff = 1.0 + 1e-6 * avn  # Effective (average) index
    else:
        # without numerical integration
        hd = Xa[2] / 1e3  # Antenna altitude
        # if (np.abs(hd-h0) > 1e-10):
        avn = (ns / (kr * (hd - h0))) * (np.exp(kr * hd) - np.exp(kr * h0))
        # else:
        #    avn = ns*np.exp(kr*h0)
        n_eff = 1.0 + 1e-6 * avn  # Effective (average) index
    return n_eff


# @njit(cache=True, fastmath=True)
@njit(**kwd)
def swf_loss(params, Xants, tants, cr=1.0, verbose=False):

    """
    Defines Chi2 by summing model residuals over antennas  (i):
    loss = \sum_i ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    """

    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st * cp, st * sp, ct])
    Xmax = -r_xmax * K
    Xmax += np.array([0.0, 0.0, groundAltitude])
    # Xmax is in the opposite direction to shower propagation.

    # Make sure Xants and tants are compatible
    if Xants.shape[0] != nants:
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    tmp = 0.0
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i, :])
        # n_average = 1.0 #DEBUG
        dX = Xants[i, :] - Xmax
        # Spherical wave front
        res = cr * (tants[i] - t_s) - n_average * np.sqrt(
            dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]
        )
        tmp += res * res

    chi2 = tmp
    if verbose:
        print("Chi2 = ", chi2)
    return chi2


# @njit(cache=True, fastmath=True)
@njit(**kwd)
def swf_grad(params, Xants, tants, cr=1.0, verbose=False):
    """
    Gradient of SWF_loss, w.r.t. theta, phi, r_xmax and t_s
    """
    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st * cp, st * sp, ct])
    Xmax = -r_xmax * K
    # Xmax += np.array(        [0.0, 0.0, groundAltitude]    )  # Xmax is in the opposite direction to shower propagation.
    # Derivatives of Xmax, w.r.t. theta, phi, r_xmax
    dK_dtheta = np.array([ct * cp, ct * sp, -st])
    dK_dphi = np.array([-st * sp, st * cp, 0.0])
    dXmax_dtheta = -r_xmax * dK_dtheta
    dXmax_dphi = -r_xmax * dK_dphi
    dXmax_drxmax = -K

    jac = np.zeros(4)
    jac_temp = np.zeros(4)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i, :])
        ## n_average = 1.0 ## DEBUG
        dX = Xants[i, :] - Xmax
        # ndX = np.linalg.norm(dX)
        ndX = np.sqrt(dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2])
        res = cr * (tants[i] - t_s) - n_average * ndX
        # Derivatives w.r.t. theta, phi, r_xmax, t_s
        coef = 2 * n_average * res / ndX
        jac[0] += coef * (
            dXmax_dtheta[0] * dX[0] + dXmax_dtheta[1] * dX[1] + dXmax_dtheta[2] * dX[2]
        )
        jac[1] += coef * (dXmax_dphi[0] * dX[0] + dXmax_dphi[1] * dX[1] + dXmax_dphi[2] * dX[2])
        jac[2] += coef * (
            dXmax_drxmax[0] * dX[0] + dXmax_drxmax[1] * dX[1] + dXmax_drxmax[2] * dX[2]
        )
        jac[3] += -2 * cr * res
    # if (verbose):
    #     print ("Jacobian = ",jac)
    return jac


# @njit
def swf_hess(params, Xants, tants, cr=1.0, verbose=False):
    """
    Hessian of SWF loss, w.r.t. theta, phi, r_xmax, t_s
    """
    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st * cp, st * sp, ct])
    Xmax = -r_xmax * K
    # Xmax += np.array(        [0.0, 0.0, groundAltitude]    )  # Xmax is in the opposite direction to shower propagation.
    # Derivatives of Xmax, w.r.t. theta, phi, r_xmax
    dK_dtheta = np.array([ct * cp, ct * sp, -st])
    dK_dphi = np.array([-st * sp, st * cp, 0.0])
    dXmax_dtheta = -r_xmax * dK_dtheta
    dXmax_dphi = -r_xmax * dK_dphi
    dXmax_drxmax = -K
