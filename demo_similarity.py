from sce import *
from scipy.sparse import load_npz
from numpy import load

print "This demo requires two data files."
print "You can download IJCNN_similarity.npz from https://ntnu.box.com/s/x6akg9r111m5bws77kt7a5zpz2tg7bun"
print "You can download IJCNN_labels.npy from https://ntnu.box.com/s/z8a5if5bmkjxxg5pvm4k3cf2okdj1e0t"

P = load_npz('IJCNN_similarity.npz')
labels = load('IJCNN_labels.npy')
P = (P + P.T > 0).astype(np.float).tocoo()
Y = sce(P, 100000)
display_visualization(Y, labels, marker_size=1)
