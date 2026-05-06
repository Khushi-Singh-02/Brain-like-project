import numpy as np

def pearson_r_per_unit(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    yt = Y_true - Y_true.mean(0)
    yp = Y_pred - Y_pred.mean(0)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0)) + 1e-12
    return num / den

def explained_variance_per_unit(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((Y_true - Y_pred) ** 2).sum(0)
    ss_tot = ((Y_true - Y_true.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot

def nc_correct_r(r: np.ndarray, nc_ev: np.ndarray) -> np.ndarray:
    nc_r = np.sqrt(np.clip(nc_ev, 1e-6, None))
    return r / nc_r

def nc_correct_ev(ev: np.ndarray, nc_ev: np.ndarray) -> np.ndarray:
    return ev / np.clip(nc_ev, 1e-6, None)

def summarise_metrics(r, r_nc, ev, ev_nc, nc_flat, nc_threshold=10.0) -> dict:
    mask = nc_flat > nc_threshold
    def _m(arr):
        if mask.sum() == 0:
            return float(np.nanmean(arr))
        return float(np.nanmean(arr[mask]))
    return {
        "pearsonr":              _m(r),
        "pearsonr_nc":           _m(r_nc),
        "explained_variance":    _m(ev),
        "explained_variance_nc": _m(ev_nc),
    }