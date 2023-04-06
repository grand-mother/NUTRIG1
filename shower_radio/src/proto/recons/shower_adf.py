"""
Estimate shower parameters with ADF model (adf_)

=================
Work in progress 
=================
"""

import numba as njit
import numpy as np

import sradio.recons.shower_spheric as swf

# Magnetic field direction (unit) vector$
B_dec = 0.0
B_inc = np.pi / 2.0 + 1.0609856522873529
Bvec = np.array([np.sin(B_inc) * np.cos(B_dec), np.sin(B_inc) * np.sin(B_dec), np.cos(B_inc)])

# Used for interpolation
n_omega_cr = 20

R_earth = 6371007.0
ns = 325
kr = -0.1218
inv_kr = 1 / kr
groundAltitude = 1086.0

kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}


@njit
def RefractionIndexAtPosition(pos):
    """

    :param pos: cartesian position
    :type pos: float (3,)
    """
    R2 = pos[0] * pos[0] + pos[1] * pos[1]
    h = (np.np.sqrt((pos[2] + R_earth) ** 2 + R2) - R_earth) / 1e3  # Altitude in km
    rh = ns * np.exp(kr * h)
    n = 1.0 + 1e-6 * rh
    return n


@njit(**kwd)
def rotation(angle, axis):
    """
    Compute rotation matrix from angle and axis coordinates,
    using Rodrigues formula
    """
    ca = np.cos(angle)
    sa = np.sin(angle)

    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    mat = np.eye(3) + sa * cross + (1.0 - ca) * np.dot(cross, cross)
    return mat


@njit(**kwd)
def der(func, x, args=[], eps=1e-7):
    """
    Forward estimate of derivative
    """
    return (func(x + eps, *args) - func(x, *args)) / eps


@njit(**kwd)
def newton(func, x0, tol=1e-7, nstep_max=100, args=[], verbose=False):
    """
    Newton method for zero finding.
    Uses forward estimate of derivative
    """
    rel_error = np.infty
    xold = x0
    nstep = 0
    while (rel_error > tol) and (nstep < nstep_max):
        x = xold - func(xold, *args) / der(func, xold, args=args)
        nstep += 1
        if verbose == True:
            print("x at iteration", nstep, "is ", x)
        rel_error = np.abs((x - xold) / xold)
        xold = x
    #    if (nstep == nstep_max):
    #        print ("Convergence not achieved in %d iterations"%nstep_max)
    return x


@njit(**kwd)
def compute_observer_position(omega, Xmax, U, K):
    """
    Given angle between shower direction (K) and line joining Xmax and observer's position,
    horizontal direction to observer's position, Xmax position and groundAltitude, compute
    coordinates of observer
    """

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U, K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega, Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat, K)
    # Compute observer's position
    t = (groundAltitude - Xmax[2]) / Dir_obs[2]
    X = Xmax + t * Dir_obs
    return X


@njit(**kwd)
def minor_equation(omega, n0, n1, alpha, delta, xmaxDist):
    """
    Compute time delay (in m)
    """
    Lx = xmaxDist
    sa = np.sin(alpha)
    saw = np.sin(
        alpha - omega
    )  # Keeping minus sign to compare to Valentin's results. Should be plus sign.
    com = np.cos(omega)
    # Eq. 3.38 p125.
    res = (
        Lx * Lx * sa * sa * (n0 * n0 - n1 * n1)
        + 2 * Lx * sa * saw * delta * (n0 - n1 * n1 * com)
        + delta * delta * (1.0 - n1 * n1) * saw * saw
    )

    return res


@njit(**kwd)
def compute_delay(omega, Xmax, Xb, U, K, alpha, delta, xmaxDist):

    X = compute_observer_position(omega, Xmax, U, K)
    # print('omega = ',omega,'X_obs = ',X)
    n0 = swf.ZHSEffectiveRefractionIndex(Xmax, X)
    # print('n0 = ',n0)
    n1 = swf.ZHSEffectiveRefractionIndex(Xb, X)
    # print('n1 = ',n1)
    res = minor_equation(omega, n0, n1, alpha, delta, xmaxDist)
    # print('delay = ',res)
    return res


@njit(**kwd)
def compute_Cerenkov(eta, K, xmaxDist, Xmax, delta, groundAltitude):

    """
    Solve for Cerenkov angle by minimizing
    time delay between light rays originating from Xb and Xmax and arriving
    at observer's position.
    eta:   azimuth of observer's position in shower plane coordinates
    K:     direction vector of shower
    Xmax:  coordinates of Xmax point
    delta: distance between Xmax and Xb points
    groundAltitude: self explanatory

    Returns:
    omega: angle between shower direction and line joining Xmax and observer's position

    """

    # Compute coordinates of point before Xmax
    Xb = Xmax - delta * K
    # Projected shower direction in horizontal plane
    nk2D = np.sqrt(K[0] * K[0] + K[1] * K[1])
    K_plan = np.array([K[0] / nk2D, K[1] / nk2D, 0.0])
    # Direction vector to observer's position in horizontal plane
    # This assumes all observers positions are in the horizontal plane
    ce = np.cos(eta)
    se = np.sin(eta)
    U = np.array([ce * K_plan[0] + se * K_plan[1], -se * K_plan[0] + ce * K_plan[1], 0.0])
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.arccos(np.dot(K, U))

    # Now solve for omega
    # Starting point at standard value acos(1/n(Xmax))
    omega_cr_guess = np.arccos(1.0 / RefractionIndexAtPosition(Xmax))
    # print("###############")
    # omega_cr = fsolve(compute_delay,[omega_cr_guess])
    omega_cr = newton(
        compute_delay, omega_cr_guess, args=(Xmax, Xb, U, K, alpha, delta, xmaxDist), verbose=False
    )
    ### DEBUG ###
    # omega_cr = omega_cr_guess
    return omega_cr


@njit(**kwd)
def ADF_loss(params, Aants, Xants, Xmax, asym_coeff=0.01, verbose=False):

    """
    Defines Chi2 by summing *amplitude* model residuals over antennas (i):
    loss = \sum_i (A_i - f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax))**2
    where the ADF function reads:

    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)

    where

    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )

    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring,
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.
    Derived parameters are:
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position
    """

    theta, phi, delta_omega, amplitude = params
    nants = Aants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([st * cp, st * sp, ct])
    K_plan = np.array([K[0], K[1]])
    KxB = np.cross(K, Bvec)
    KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K, KxB)
    KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB, KxKxB, K))
    #
    XmaxDist = (groundAltitude - Xmax[2]) / K[2]
    # print('XmaxDist = ',XmaxDist)
    asym = asym_coeff * (1.0 - np.dot(K, Bvec) ** 2)  # Azimuthal dependence, in \sin^2(\alpha)
    #
    # Make sure Xants and tants are compatible
    if Xants.shape[0] != nants:
        print("Shapes of Aants and Xants are incompatible", Aants.shape, Xants.shape)
        return None

    # Precompute an array of Cerenkov angles to interpolate over (as in Valentin's code)
    omega_cerenkov = np.zeros(2 * n_omega_cr + 1)
    xi_table = np.arange(2 * n_omega_cr + 1) / n_omega_cr * np.pi
    for i in range(n_omega_cr + 1):
        omega_cerenkov[i] = compute_Cerenkov(xi_table[i], K, XmaxDist, Xmax, 2.0e3, groundAltitude)
    # Enforce symmetry
    omega_cerenkov[n_omega_cr + 1 :] = (omega_cerenkov[:n_omega_cr])[::-1]

    # Loop on antennas
    tmp = 0.0
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i, :] - Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat, dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1], dX_sp[0])
        omega = np.arccos(np.dot(K, dX) / l_ant)
        # vector in the plane defined by K and dX, projected onto
        # horizontal plane
        val_plan = np.array([dX[0] / l_ant - K[0], dX[1] / l_ant - K[1]])
        # Angle between k_plan and val_plan
        xi = np.arccos(np.dot(K_plan, val_plan) / np.linalg.norm(K_plan) / np.linalg.norm(val_plan))

        # omega_cr = compute_Cerenkov(xi,K,XmaxDist,Xmax,2.0e3,groundAltitude)
        # Interpolate to save time
        omega_cr = np.interp(xi, xi_table, omega_cerenkov)
        # omega_cr = 0.015240011539221762
        # omega_cr = np.arccos(1./RefractionIndexAtPosition(Xmax))
        # print ("omega_cr = ",omega_cr)

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2] / l_ant) * delta_omega
        # Distribution
        adf = (
            amplitude
            / l_ant
            / (1.0 + 4.0 * (((np.tan(omega) / np.tan(omega_cr)) ** 2 - 1.0) / width) ** 2)
        )
        adf *= 1.0 + asym * np.cos(eta)  #
        # Chi2
        tmp += (Aants[i] - adf) ** 2

    chi2 = tmp
    if verbose:
        print("params = ", np.rad2deg(params[:2]), params[2:], " Chi2 = ", chi2)
    return chi2


@njit(**kwd)
def log_ADF_loss(params, Aants, Xants, Xmax, asym_coeff=0.01, verbose=False):

    return np.log10(ADF_loss(params, Aants, Xants, Xmax, asym_coeff=asym_coeff, verbose=verbose))
