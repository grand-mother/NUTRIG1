"""
From: https://github.com/simon-prunet/ADFRecons
"""

import timeit

import numpy as np
import scipy.optimize as so

from sradio.recons.shower_plane import pwf_loss, pwf_grad, pwf_hess
from sradio.recons.shower_spheric import swf_loss, swf_grad
from sradio.num.const import C_LIGHT_MS


REF_POS = "tests/recons/ref_recons_coord_antennas.txt"
REF_EVENT = "tests/recons/ref_recons_coinctable.txt"


def read_ref_data():
    # assume REF_POS and REF_TIME in same order
    # read only x, y,z
    pos = np.loadtxt(REF_POS, usecols=(1, 2, 3))
    event = np.loadtxt(REF_EVENT, usecols=(2, 3))
    t_evt = event[:, 0]
    val_evt = event[:, 1]
    print(pos.shape, t_evt.shape, val_evt.shape)
    return pos, t_evt, val_evt


def time_pwf(grad=False, hess=False, verbose=False, number=1000):

    current_recons = 0
    pos, t_evt, val_evt = read_ref_data()
    args = (pos.copy(), t_evt.copy() * C_LIGHT_MS)
    params_in = np.array([3 * np.pi / 4.0, np.pi])
    if grad:
        total_time = timeit.timeit(
            lambda: so.minimize(pwf_loss, params_in, jac=pwf_grad, args=args, method="BFGS"),
            number=number,
        )
    elif hess:
        total_time = timeit.timeit(
            lambda: so.minimize(
                pwf_loss, params_in, jac=pwf_grad, hess=pwf_hess, args=args, method="Newton-CG"
            ),
            number=number,
        )
    else:
        total_time = timeit.timeit(
            lambda: so.minimize(pwf_loss, params_in, args=args, method="BFGS"), number=number
        )

    print("Time to minimize loss = %.2f" % (total_time / number * 1000), " (ms)")

    if verbose:
        args = (pos.copy(), t_evt.copy() * C_LIGHT_MS, 1, True)

    if grad:
        res = so.minimize(pwf_loss, params_in, jac=pwf_grad, args=args, method="BFGS")
    elif hess:
        res = so.minimize(
            pwf_loss, params_in, jac=pwf_grad, hess=pwf_hess, args=args, method="Newton-CG"
        )
    else:
        res = so.minimize(pwf_loss, params_in, args=args, method="BFGS")
    params_out = res.x
    print("Best fit parameters = ", *np.rad2deg(params_out))
    print("Chi2 at best fit = ", pwf_loss(params_out, *args))
    print(res)

    return res


def time_swf(grad=False, verbose=False, number=100):

    current_recons = 0
    theta_in, phi_in = np.deg2rad(np.array([116, 270]))
    pos, t_evt, val_evt = read_ref_data()
    args = (pos.copy(), t_evt.copy() * C_LIGHT_MS)
    # Guess parameters
    bounds = [
        [np.deg2rad(theta_in - 1), np.deg2rad(theta_in + 1)],
        [np.deg2rad(phi_in - 1), np.deg2rad(phi_in + 1)],
        [
            -15.6e3 - 12.3e3 / np.cos(np.deg2rad(theta_in)),
            -6.1e3 - 15.4e3 / np.cos(np.deg2rad(theta_in)),
        ],
        [6.1e3 + 15.4e3 / np.cos(np.deg2rad(theta_in)), 0],
    ]
    params_in = np.array(bounds).mean(axis=1)
    print("params_in = ", params_in)

    if grad:
        total_time = timeit.timeit(
            lambda: so.minimize(swf_loss, params_in, jac=swf_grad, args=args, method="BFGS"),
            number=number,
        )
    else:
        total_time = timeit.timeit(
            lambda: so.minimize(swf_loss, params_in, args=args, method="BFGS"), number=number
        )

    print("Time to minimize loss = %.2f" % (total_time / number * 1000), " (ms)")

    if verbose:
        args = (pos.copy(), t_evt.copy() * C_LIGHT_MS, 1, True)
    res = so.minimize(swf_loss, params_in, args=args, method="BFGS")
    params_out = res.x
    print("Best fit parameters = ", *np.rad2deg(params_out[:2]), *params_out[2:])
    print("Chi2 at best fit = ", swf_loss(params_out, *args))

    return


# def time_ADF(verbose=True,number=100):
#
#     # RUN FIRST python recons.py 1 '../Chiche'
#     # Read guess for theta, phi from plane wave
#     fid_angles = open(data_dir+'Rec_plane_wave_recons_py.txt')
#     l = fid_angles.readline().strip().split()
#     theta_in,phi_in = np.float(l[2]),np.float(l[4])
#
#     an = antenna_set('../Chiche/coord_antennas.txt')
#     co = coincidence_set('../Chiche/Rec_coinctable.txt',an)
#     current_recons = 0
#     # Read guess for Xmax from spherical wave
#     fid_xmax = open(data_dir+'Rec_sphere_wave_recons_py.txt')
#     l = fid_xmax.readline().strip().split()
#     Xmax = np.array([float(l[4]),float(l[5]),float(l[6])])
#     # Init parameters. Make better guess for amplitude from max of peak amplitudes
#     bounds = [[np.deg2rad(theta_in-1),np.deg2rad(theta_in+1)],
#              [np.deg2rad(phi_in-1),np.deg2rad(phi_in+1)],
#              [0.1,3.0],
#              [1e6,1e10]]
#     params_in = np.array(bounds).mean(axis=1) # Central values
#     ## Refine guess for amplitude, based on maximum of peak values ##
#     lant = (groundAltitude-Xmax[2])/np.cos(np.deg2rad(theta_in))
#     params_in[3] = co.peak_amp_array[current_recons,:].max() * lant
#     print ('params_in = ',params_in)
#
#     args=(co.peak_amp_array[current_recons,:],co.antenna_coords_array[current_recons,:],Xmax)
#     total_time = timeit.timeit(lambda: so.minimize(ADF_loss,params_in,args=args,method='BFGS'),number=number)
#
#     print ("Time to minimize loss = %.2f"%(total_time/number*1000), " (ms)")
#
#     if verbose:
#         args=(co.peak_amp_array[current_recons,:],co.antenna_coords_array[current_recons,:],Xmax,0.01,True)
#
#     res = so.minimize(ADF_loss,params_in,args=args,method='BFGS')
#     params_out = res.x
#     print ("Best fit parameters = ",*np.rad2deg(params_out[:2]),*params_out[2:])
#     print ("Chi2 at best fit = ",ADF_loss(params_out,*args))
#
#
#     return (res)
