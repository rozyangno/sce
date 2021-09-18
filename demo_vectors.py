from sklearn.datasets import load_digits
from sklearn.neighbors import kneighbors_graph
from sce import *

X, labels = load_digits(return_X_y=True)
P = kneighbors_graph(X, 10)
P = (P + P.T > 0).astype(np.float).tocoo()
Y = sce(P, 1000)
display_visualization(Y, labels)
