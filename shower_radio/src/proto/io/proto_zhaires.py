"""
Created on 28 mars 2023

@author: jcolley
"""
import pprint

import matplotlib.pyplot as plt

import sradio.io.shower.zhaires_txt as zh


tt_sry = """
>>>>
>>>>
>>>>        AAAAA    IIII   RRRRRR    EEEEEEE    SSSSS       \
>>>>       AAAAAAA   IIII   RRRRRRR   EEEEEEE   SSSSSSS       A
>>>>       AA   AA    II    RR   RR   EE        SS   SS        I
>>>>       AAAAAAA    II    RRRRRR    EEEE       SSS            R-shower
>>>>       AAAAAAA    II    RRRRR     EEEE         SSS          Extended
>>>>       AA   AA    II    RR  RR    EE        SS   SS         Simulations
>>>>       AA   AA   IIII   RR   RR   EEEEEEE   SSSSSSS         |
>>>>       AA   AA   IIII   RR   RR   EEEEEEE    SSSSS         / \
>>>>
>>>>
>>>>       Departamento de Fisica, Universidad de La Plata, ARGENTINA.
>>>>
>>>>
>>>> This is AIRES version 19.04.00 (24/Apr/2019)
>>>> With ZHAireS version 1.0.28b (12/Apr/2020) extension
>>>> (Compiled by tueros@cca004, date: 15/Apr/2020) **
>>>> USER: tueros, HOST: ccwsge0459.in2p3.fr, DATE: 08/Jul/2020
>>>>   
    BASIC PARAMETERS:
                                Site: Dunhuang
                                      (Lat:  40.00 deg. Long:   93.10 deg.)
                                Date: 13/May/2021
                                
                                
                    Primary particle: Proton
                      Primary energy: 3.9808 EeV
                Primary zenith angle:    74.76 deg
               Primary azimuth angle:     0.00 deg
              Zero azimuth direction: Local magnetic north
                     Thinning energy: 1.0000E-05 Relative
     (D)          Injection altitude: 100.00 km (1.2829228E-03 g/cm2)
                     Ground altitude: 1.0860 km (906.9869 g/cm2)
           First obs. level altitude: 100.000 km (1.2829228E-03 g/cm2)
            Last obs. level altitude: 1.0900 km (906.5477 g/cm2)
          Obs. levels and depth step:        510     1.781 g/cm2
                   Geomagnetic field: Intensity: 55.997 uT
                                      I:   60.79 deg. D:    0.36 deg
     (D)         Table energy limits: 10.000 MeV to 2.9856 EeV
                 Table radial limits: 100.00 m  to 20.000 km

 Sl. depth of max. (g/cm2):   749.246     0.00     0.00   749.25   749.25
 Charged pcles. at maximum:  91.57902   0.0000   0.0000  91.5790  91.5790
              (Times 10^6)

  Geomagnetic field: Intensity: 55.997 uT
                                      I:   60.79 deg. D:    0.36 deg           
 
                             Altitude   Dist.     x        y        z
      Location of max.(Km):    11.808    40.34    38.92     0.00    11.69

     The fits were done with the Levenberg-Marquardt
     nonlinear least-squares fitt_srying algorithm, modelling

   
      Time bin size:  0.50ns

>>>>

    """

path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_LH_EPLHC_Proton_3.98_84.5_180.0_2"
path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_0.12_74.8_0.0_1"
)
path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_0.567_74.8_0.0_1"
)


def test_sry():
    sry = zh.ZhairesSummaryFileVers28(str_sry=tt_sry)
    pprint.pprint(sry.get_dict())
    print(sry.is_ok())


def try_zhaires_convert():
    simu_zh = zh.ZhairesSingleEventText(path_simu)
    simu_zh.read_all()
    pprint.pprint(simu_zh.d_info)
    simu = simu_zh.get_object_3dtraces()
    simu.plot_footprint_val_max()
    simu.plot_footprint_time_max()
    simu.plot_footprint_4d()


if __name__ == "__main__":
    # test_version()
    # test_xmax()
    # test_zenith()
    # test_sry()
    try_zhaires_convert()
    plt.show()
