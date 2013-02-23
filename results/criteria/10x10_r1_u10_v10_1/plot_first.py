#!/usr/bin/env python3

import os
import sys
sys.path.append('../../..')

from plot_results import *

def plot_firsts(filename, outdir, size=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig_width_pt = 120.96277  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    #golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    #fig_height = fig_width*golden_mean      # height in inches
    fig_size = [fig_width, fig_width * .8]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'savefig.dpi': 150}
    pylab.rcParams.update(params)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    res = load_results(filename)
    for k, v in res.items():
        if k.startswith('_'):
            continue
        assert v[0][3] is None

        i, j = v[1][2]
        vals = v[1][3]
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        diff = vmax - vmin

        plt.figure()
        ax = plt.gca()
        #plt.xticks([])
        #plt.yticks([])
        #plt.xlim(.5, 9.5)
        #plt.ylim(.5, 9.5)
        divider = make_axes_locatable(ax)
        im = ax.matshow(vals, vmin=vmin - .1 * diff,
                              vmax=vmax,
                              cmap=plt.cm.YlOrRd,
                              origin='upper')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)

        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax.scatter(j, i, marker='x', c='black', s=10)  # s=15)
        plt.savefig(os.path.join(outdir, k + '.pdf'), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    plot_firsts(sys.argv[1], sys.argv[2])
