#!/usr/local/bin/python27

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pylab as pl
import numpy as np
import scipy as sp
import os, sys
from subprocess import check_call
from random import sample
import json

suffixes = ["_96000_" + str(n) for n in [-1, -2, -10, 0]]
for path_suffix in ["_96000_-2"]:

    data = json.load(open("./embeddings" + path_suffix + ".json"))
    colors = json.load(open("./primarycolors.json"))
    vs = data.values()
    ks = data.keys()
    x = np.array([-t['x'] for t in vs])
    y = np.array([-t['y'] for t in vs])
    c = np.array([colors[t['idx']] for t in vs])
    w = np.array([t for t in ks])
    frac = np.random.rand(len(w))
    frac[(-len(w) // 2):] = 1



    pd = .0256
    xbds = np.array([x.min(), x.max()]) + np.array([-pd, pd])
    ybds = -np.array([y.max(), y.min()]) + np.array([-pd, pd])



    seed_words = {'the', 'bose_einstein_condensation', 'quantum_spin_hall_effect', 'parsed', 'qubit', 'spiking_neurons',
                  'quark', 'black_branes', 'vision_tasks', 'unfortunate', 'graphene', 'plotstyle', 'facile', 'genetic',
                  'groups', 'subschemes', 'finance', 'scholarly', 'logistic', 'smith', 'inflation', 'telescope',
                  'polymorphic', 'divisibility', 'holomorphic', 'paraboloid', 'hay', 'bremen', 'sunspot', 'liquid',
                  'nanorod', 'milky_way_galaxy', 'viscosity', 'maser', 'halo', 'metal_rich', 'supersymmetry', 'wi_fi',
                  'checksum', 'igor', 'cornell_university', 'portal', 'kmeans', 'multistep', 'optimization'}

    # no daniela
    dir_base = os.path.join(os.getcwd(), "zoom" + path_suffix)
    zooms = 8
    spacings = np.linspace(0, 1, zooms + 1) ** 6
    spacings[0] = -1
    spacings[1] = -1
    for zoom in range(zooms + 1):
        print("ZOOM", zoom, spacings[zoom])
        divs = 1 << zoom
        xscale = np.diff(xbds) / divs
        yscale = np.diff(ybds) / divs
        dir_zoom = os.path.join(dir_base, str(zoom))
        check_call(['mkdir', '-p', dir_zoom])
        for i in range(divs):
            dir_slice = os.path.join(dir_zoom, str(i))
            check_call(['mkdir', '-p', dir_slice])
            for j in range(divs):
                # print(i,j)
                x0 = xscale * i + xbds[0]
                y0 = ybds[1] - (yscale * (j + 1))
                mask = (x > x0 - xscale) & (x < x0 + 2 * xscale) & (-y > y0 - yscale) & (-y < y0 + 2 * yscale)

                xs = x[mask]
                ys = y[mask]
                cs = c[mask]
                ws = w[mask]
                fracs = frac[mask]

                fig, axs = pl.subplots(figsize=(10, 10))
                axs.scatter(xs, -ys, c=cs, edgecolors="#eeeeee", lw=0.0, s=5 * divs, alpha=0.5)
                axs.set_xlim([x0, x0 + xscale])
                axs.set_ylim([y0, y0 + yscale])
                axs.axis('off')
                axs.grid('off')
                pl.subplots_adjust(0, 0, 1, 1, 0, 0)

                for w0, x0, y0, frac0 in zip(ws, xs, ys, fracs):
                    if zoom == zooms or (zoom > 2 and frac0 <= spacings[zoom]) \
                            or (zoom >= 2 and w0 in seed_words):
                        txt = axs.text(x0, -y0, w0, fontsize=40, alpha=0.72)
                        txt.set_bbox(dict(color='white', alpha=0.72 if zoom > 2 else .25,  # was .8
                                          edgecolor='white'))  # was .8

                pl.savefig(os.path.join(dir_slice, str(j) + ".png"), dpi=25.6)
                pl.close('all')

        if len(sys.argv) > 1 and sys.argv[1] == 's' and zoom == 2: break
