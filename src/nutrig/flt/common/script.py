'''


'''

import matplotlib.pyplot as plt

import nutrig.flt.common.tools_trace as tt
import nutrig.data_api.du.trend.io as tio
from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg

import proto.simu.proto_simu_du as simu
from sradio.basis.traces_event import Handling3dTracesOfEvent
from sradio.io.shower import zhaires_base as zbase

# CCIN2P3
PATH_DATA = "/sps/trend/slecoz/"
# local
PATH_DATA = "/home/jcolley/projet/nutrig_wk/data/"

logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("debug", log_stdout=True)


def proto_extremas_relatif(file_simu):
    zsimu = ZhairesMaster(file_simu)
    events = zsimu.get_object_3dtraces()
    r1, _, _ = tt.extract_extreltraces_3d(events.traces)
    print(r1)
    events.plot_footprint_val_max()
    tt.plot_extra_relatif(r1)
    

def proto_extremas_relatif_volt(file_simu):
    efield, volt, s_info = simu.proto_simu_vout(file_simu, "/home/jcolley/temp/grand/volt.asdf")
    #volt.remove_traces_low_signal(2000)
    assert isinstance(volt, Handling3dTracesOfEvent)
    assert isinstance(efield, Handling3dTracesOfEvent)
    efield.plot_footprint_val_max()
    volt.plot_footprint_val_max()
    r1, _, _ = tt.extract_extreltraces_3d(volt.traces)
    x_dir = volt.get_pos_direction(zbase.get_simu_xmax(s_info), True)
    print(r1)
    volt.network.plot_footprint_1d(r1[:, 1], "extremum relatif", volt, "lin", "unitless")
    tt.plot_extra_relatif(r1, ((x_dir[:, 0] + 180) % 360))    
    tt.plot_extra_relatif(r1, x_dir[:, 1])

    
if __name__ == "__main__":
    # file 
    file_trace = "MLP6hybrid/MLP6hybrid_selected.bin"
    file_trace = "MLP6/MLP6_selected.bin"
    file_trace = "MLP6/MLP6_transient.bin"
    file_trace = "MLhybrid_selected.bin"
    root_path_zhaires = '/home/jcolley/projet/grand_wk/data/zhaires/'
    file_efield = root_path_zhaires + 'GP300_Xi_Sib_Iron_3.85_64.4_256.1_19063/'
    file_efield = root_path_zhaires + 'GP300_Iron_1.58_85.0_164.41_12'
    # no voltage
    file_efield = root_path_zhaires + "GP300_Proton_0.631_87.1_132.07_15"
    file_efield = root_path_zhaires + "Stshp_Iron_3.98_87.1_0.0_1"
    file_efield = root_path_zhaires + "GP300_Xi_Sib_Iron_0.17_84.7_188.6_24023"
    signalfilename = PATH_DATA + file_trace
    data_title = signalfilename.split('/')[-1]
    print(data_title)
    # 
    if False:
        signal = tio.read_trace_trend(signalfilename)
        def_sig = tt.PulsExtractor(signal[20:30,:])
        ret = def_sig.extract_pulse_2()
        print(ret)
    # proto_extremas_relatif('/home/jcolley/projet/grand_wk/data/zhaires/GP300_Iron_1.58_85.0_164.41_12')
    # proto_extremas_relatif('/home/jcolley/projet/grand_wk/data/zhaires/GP300_Proton_0.631_87.1_132.07_15')
    # proto_extremas_reltif('/home/jcolley/projet/grand_wk/data/zhaires/Stshp_Iron_3.98_87.1_0.0_1')
    proto_extremas_relatif_volt(file_efield)
    #
    #
    plt.show()
