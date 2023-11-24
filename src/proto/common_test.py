'''
Created on 8 nov. 2023

@author: jcolley
'''

import nutrig.flt.common.tools_trace as nut
import numpy as np
import matplotlib.pyplot as plt


def use_PulsExtractor3D():
    tr_ran = np.random.normal(0, 20, 10 * 1024 * 3).reshape(10, 3, 1024)
    p3d = nut.PulsExtractor3D(tr_ran)
    p3d.extract_fixed_with_extremum()
    p3d.plot_pulse_selected()


if __name__ == '__main__':
    use_PulsExtractor3D()
    plt.show()
