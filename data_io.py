import h5py
import numpy as np
def list_h5_layers(path: str) -> list[str]:
    layers = []
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name != "ids":
            layers.append(name)
    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    return sorted(layers)

def h5_indexed_read(path: str, dataset_key: str, indices: np.ndarray) -> np.ndarray:
    sort_order = np.argsort(indices)
    restore_order = np.argsort(sort_order)
    sorted_idx = indices[sort_order]
    with h5py.File(path, "r") as f:
        data = f[dataset_key][sorted_idx.tolist(), :]
    return data[restore_order]

def make_index_map(feat_path: str) -> dict:
    with h5py.File(feat_path, "r") as f:
        ids = f["ids"][:]
    return {v: i for i, v in enumerate(ids)}

def get_feat_indices(feat_path: str, stimulus_ids: np.ndarray) -> np.ndarray:
    idx_map = make_index_map(feat_path)
    return np.array([idx_map[s] for s in stimulus_ids])