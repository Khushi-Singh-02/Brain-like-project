import numpy as np
import torch
import time
from metrics import *

def fit_ridge_encoding_closed_form(X_train, Y_train, DEVICE, val_frac=0.15, seed=42, alpha_grid=None, **kwargs):
    if alpha_grid is None:
        alpha_grid = np.logspace(0, 7, 15)

    rng     = np.random.default_rng(seed)
    n       = X_train.shape[0]
    n_val   = max(1, int(n * val_frac))
    val_idx = rng.choice(n, n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(n), val_idx)

    Xtr, Ytr = X_train[tr_idx], Y_train[tr_idx]
    Xvl, Yvl = X_train[val_idx], Y_train[val_idx]

    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(DEVICE)
    mean  = Xtr_t.mean(0, keepdim=True)
    std   = Xtr_t.std(0, keepdim=True).clamp(min=1e-8)
    Xtr_t = (Xtr_t - mean) / std
    Xvl_t = (torch.from_numpy(Xvl.astype(np.float32)).to(DEVICE) - mean) / std

    # ─── CENTER Y ───
    Ytr_t  = torch.from_numpy(Ytr.astype(np.float32)).to(DEVICE)
    y_mean = Ytr_t.mean(0, keepdim=True)             # (1, C)
    Ytr_c  = Ytr_t - y_mean                           # centered for fitting
    # ────────────────

    N = Xtr_t.shape[0]
    K  = Xtr_t @ Xtr_t.T
    Kv = Xvl_t @ Xtr_t.T

    eigvals, U = torch.linalg.eigh(K)
    UtY = U.T @ Ytr_c                                 # use centered Y
    KvU = Kv @ U

    best_alpha, best_score = alpha_grid[0], -np.inf
    for alpha in alpha_grid:
        t0 = time.time()
        inv = 1.0 / (eigvals + float(alpha))
        Yv_hat = (KvU * inv) @ UtY + y_mean           # ADD MEAN BACK
        score = float(np.nanmean(pearson_r_per_unit(Yvl, Yv_hat.cpu().numpy())))
        marker = " ◀ best so far" if score > best_score else ""
        print(f"        α={alpha:.1e}  val_r={score:.4f}{marker}  ({time.time()-t0:.2f}s)")
        if score > best_score:
            best_score, best_alpha = score, alpha

    del Xtr_t, Xvl_t, Ytr_t, Ytr_c, K, Kv, eigvals, U, UtY, KvU
    torch.cuda.empty_cache()

    # Refit on full train
    X_full_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    mean_f   = X_full_t.mean(0, keepdim=True)
    std_f    = X_full_t.std(0, keepdim=True).clamp(min=1e-8)
    X_full_t = (X_full_t - mean_f) / std_f

    Y_full_t  = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    y_mean_f  = Y_full_t.mean(0, keepdim=True)        # (1, C) — full-train Y mean
    Y_full_c  = Y_full_t - y_mean_f

    N_full = X_full_t.shape[0]
    gamma  = torch.linalg.solve(
        X_full_t @ X_full_t.T + float(best_alpha) * torch.eye(N_full, device=DEVICE),
        Y_full_c,                                     # centered
    )
    beta = X_full_t.T @ gamma

    del X_full_t, Y_full_t, Y_full_c, gamma
    torch.cuda.empty_cache()

    return beta, mean_f, std_f, y_mean_f, best_alpha   # ← NOW RETURNS y_mean_f TOO


def fit_ridge_encoding_adam(X_train, Y_train, DEVICE, val_frac=0.15, seed=42,
                       alpha_grid=None, n_epochs=300, lr=1e-3,
                       warm_start=True, verbose=True, **kwargs):
    """Iterative ridge via Adam."""
    if alpha_grid is None:
        alpha_grid = np.logspace(0, 7, 15)

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    n_val = max(1, int(n * val_frac))
    val_idx = rng.choice(n, n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(n), val_idx)

    Xtr = torch.from_numpy(X_train[tr_idx].astype(np.float32)).to(DEVICE)
    Xvl = torch.from_numpy(X_train[val_idx].astype(np.float32)).to(DEVICE)
    Ytr = torch.from_numpy(Y_train[tr_idx].astype(np.float32)).to(DEVICE)
    Yvl_np = Y_train[val_idx].astype(np.float32)

    mean = Xtr.mean(0, keepdim=True)
    std  = Xtr.std(0, keepdim=True).clamp(min=1e-8)
    Xtr  = (Xtr - mean) / std
    Xvl  = (Xvl - mean) / std

    y_mean = Ytr.mean(0, keepdim=True)
    Ytr_c  = Ytr - y_mean

    D, C = Xtr.shape[1], Ytr.shape[1]

    best_alpha, best_score, best_beta = alpha_grid[0], -np.inf, None
    beta = torch.zeros(D, C, device=DEVICE, requires_grad=True)

    for alpha in alpha_grid:
        t0 = time.time()
        if not warm_start:
            beta = torch.zeros(D, C, device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([beta], lr=lr)

        for ep in range(n_epochs):
            opt.zero_grad()
            Y_pred = Xtr @ beta
            # sum-of-squares ridge loss → α range matches closed-form grid
            loss = 0.5 * ((Y_pred - Ytr_c) ** 2).sum() \
                 + 0.5 * float(alpha) * (beta ** 2).sum()
            if ep % 50 == 0:
                print(f"          ep {ep:3d}  loss={loss.item():.3e}  ||β||={beta.norm().item():.2f}")
            loss.backward()
            opt.step()

        with torch.no_grad():
            Yv_hat = (Xvl @ beta + y_mean).cpu().numpy()
        score = float(np.nanmean(pearson_r_per_unit(Yvl_np, Yv_hat)))
        marker = " ◀ best so far" if score > best_score else ""
        if verbose:
            print(f"        α={alpha:.1e}  val_r={score:.4f}{marker}  "
                  f"({time.time()-t0:.2f}s)")
        if score > best_score:
            best_score, best_alpha = score, alpha
            best_beta = beta.detach().clone()

    del Xtr, Xvl, Ytr, Ytr_c, beta
    torch.cuda.empty_cache()

    # Refit on full train (tr + val) at the chosen α, warm-start from best_beta
    X_full = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    mean_f = X_full.mean(0, keepdim=True)
    std_f  = X_full.std(0, keepdim=True).clamp(min=1e-8)
    X_full = (X_full - mean_f) / std_f

    Y_full   = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    y_mean_f = Y_full.mean(0, keepdim=True)
    Y_full_c = Y_full - y_mean_f

    beta_final = best_beta.clone().requires_grad_(True)
    opt = torch.optim.Adam([beta_final], lr=lr)
    for ep in range(n_epochs):
        opt.zero_grad()
        Y_pred = X_full @ beta_final
        loss = 0.5 * ((Y_pred - Y_full_c) ** 2).sum() \
             + 0.5 * float(best_alpha) * (beta_final ** 2).sum()
        loss.backward()
        opt.step()

    beta_out = beta_final.detach()
    del X_full, Y_full, Y_full_c, beta_final
    torch.cuda.empty_cache()

    return beta_out, mean_f, std_f, y_mean_f, best_alpha


def eval_encoding(beta, mean, std, y_mean, X_test, Y_test, nc_flat, DEVICE, nc_scale=100.0, nc_threshold=10.0):
    # Assuming DEVICE is passed or defined globally in the module
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(DEVICE)
    X_test_t = (X_test_t - mean) / std
    with torch.no_grad():
        Y_pred = (X_test_t @ beta + y_mean).cpu().numpy()

    nc_scaled = nc_flat / nc_scale
    r     = pearson_r_per_unit(Y_test, Y_pred)
    ev    = explained_variance_per_unit(Y_test, Y_pred)
    r_nc  = nc_correct_r(r, nc_scaled)
    ev_nc = nc_correct_ev(ev, nc_scaled)

    summary = summarise_metrics(r, r_nc, ev, ev_nc, nc_flat, nc_threshold=nc_threshold)
    per_channel = {"r": r, "ev": ev, "r_nc": r_nc, "ev_nc": ev_nc}
    return summary, Y_pred, per_channel