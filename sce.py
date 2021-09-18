import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import random


def display_visualization(ycoords, labels=None, marker_size=10):
    n = ycoords.shape[0]
    if labels is None:
        labels = np.ones(n)
    nc = np.max(labels) + 1
    fig = plt.figure(figsize=(8, 8), dpi=120)
    colors = plt.cm.jet(np.linspace(0, 1, nc))
    for ci in range(nc):
        ind = labels == ci
        plt.scatter(ycoords[ind, 0], ycoords[ind, 1], marker='.', s=marker_size, color=colors[ci])
    plt.axis('off')
    fig.tight_layout()
    plt.show()


def sce(sim_mat, max_iter=10**5, alpha=0.5, weights=None, ycoords0=None, eta0=1,
        constant_eta=False, n_repu_samp=1, block_size=128, block_count=128, random_seed=0):

    random.seed(random_seed)
    np.random.seed(random_seed)

    nn = sim_mat.shape[0]
    ne = sim_mat.nnz

    fsim = tempfile.NamedTemporaryFile(delete=False)
    fsim.write(np.array(nn).astype(np.uint64).tobytes())
    fsim.write(np.array(ne).astype(np.uint64).tobytes())
    fsim.write(sim_mat.row.astype(np.uint64).tobytes())
    fsim.write(sim_mat.col.astype(np.uint64).tobytes())
    fsim.write(sim_mat.data.astype(np.double).tobytes())
    fsim.close()

    if weights is None:
        weights = np.ones(nn)
    fweights = tempfile.NamedTemporaryFile(delete=False)
    fweights.write(weights.astype(np.double).tobytes())
    fweights.close()

    if ycoords0 is None:
        ycoords0 = np.random.randn(nn, 2)
    fycoords0 = tempfile.NamedTemporaryFile(delete=False)
    fycoords0.write(ycoords0.astype(np.single).tobytes())
    fycoords0.close()

    fycoords = tempfile.NamedTemporaryFile(delete=False)
    fycoords.close()

    if not constant_eta:
        constant_eta = 0
    else:
        constant_eta = 1

    prog = 'sce'
    cmd_str = "./%s 1 %s %s %s %s %d %f %d %d %d %f %d" % \
              (prog, fsim.name, fycoords.name, fweights.name, fycoords0.name,
               max_iter, eta0, n_repu_samp, block_size, block_count, alpha, constant_eta)
    os.system(cmd_str)

    ycoords = np.loadtxt(fycoords.name)

    os.remove(fsim.name)
    os.remove(fweights.name)
    os.remove(fycoords.name)
    os.remove(fycoords0.name)

    return ycoords


