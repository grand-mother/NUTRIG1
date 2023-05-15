'''
Created on 15 mai 2023

@author: jcolley
'''
import pprint
import matplotlib.pylab as plt
from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg

G_path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
)
#G_path_simu = "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"
G_path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
#G_path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_LH_EPLHC_Proton_3.98_84.5_180.0_2"
#
# Logger
#
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("debug", log_stdout=True)

def test_fit_polar(f_simu):
    f_zh = ZhairesMaster(f_simu)
    i_sim = f_zh.get_simu_info()
    pprint.pprint(f_zh.get_simu_info())
    evt = f_zh.get_object_3dtraces()
    #evt.network.name += f"\nXmax dist {i_sim['x_max']['dist']:.1f}km, zenith angle: {i_sim['shower_zenith']:.1f}deg"
    evt.plot_footprint_val_max()
    #a_pol  = evt.get_polar_vec()
    #print(a_pol)
    evt.plot_polar_check_fit()

if __name__ == '__main__':
    test_fit_polar(G_path_simu)
    plt.show()