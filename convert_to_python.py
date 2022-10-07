import numpy as np
import h5py
from scipy.sparse import csc_matrix, coo_matrix


def convert_dataset(dataset):
    P_name = f"P_{dataset}"
    fmat = f"{P_name}.mat"
    fnpz = f"{P_name}.npz"

    with h5py.File(fmat) as f:
        P_data = np.squeeze(f['P']['data'][()].T)
        P_ir = np.squeeze(f['P']['ir'][()].T)
        P_jc = np.squeeze(f['P']['jc'][()].T)
        P = csc_matrix((P_data, P_ir, P_jc)).tocoo()
        if "/C" in f:
            C = np.squeeze(f['C'][()].T).astype(int)
        else:
            C = None
        if "/nc" in f:
            nc = np.squeeze(f['nc'][()].T)
        else:
            nc = None

    np.savez(fnpz, C=C, P=P, nc=nc)


convert_dataset("SHUTTLE")
convert_dataset("MNIST")
convert_dataset("IJCNN")
convert_dataset("TOMORADAR")
convert_dataset("FLOWCYTOMETRY")
convert_dataset("HIGGS")

