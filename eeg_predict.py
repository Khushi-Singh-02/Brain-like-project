import os
import time
import h5py
import numpy as np
import pandas as pd
import torch

from metrics import *
from encoding import *  # Make sure your closed-form fit_ridge_encoding is in here!
from data_io import *

def run_eeg_pipeline(
    subject: str,
    roi: str,
    rsa_fn,
    cka_fn,
    DEVICE,
    data_dir: str = "/shared/NX-414/data/",
    feat_dir: str = "/shared/NX-414/extracted_features/",
    model_a: str = "adv_resnet152_imagenet_full_ffgsm_eps-1_alpha-125-ep10_seed-0",
    model_b: str = "Qwen3-VL-2B-Instruct",
    n_train: int = None,
    n_test: int = None,
    n_layers: int = None,
    alpha_grid: np.ndarray = None,
    nc_scale: float = 100.0,
    nc_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Runs the full linear encoding and RSA/CKA pipeline for a specific EEG subject/ROI.
    Uses closed-form ridge regression and flattens/unflattens the Time dimension.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(0, 7, 15)

    target_key = f"{subject}/{roi}"
    print(f"\n{'='*60}\n🚀 STARTING EEG PIPELINE: {target_key}\n{'='*60}")
    wall_start = time.time()

    # ---------------------------------------------------------
    # 1. LOAD NEURAL DATA & NOISE CEILING
    # ---------------------------------------------------------
    eeg_path = os.path.join(data_dir, "things_eeg2.h5")
    with h5py.File(eeg_path, "r") as f:
        stim_ids_train = f["train/stimulus_ids"][:]
        stim_ids_test  = f["test/stimulus_ids"][:]
        
        if n_train: stim_ids_train = stim_ids_train[:n_train]
        if n_test:  stim_ids_test  = stim_ids_test[:n_test]
        
        # Load 3D arrays: (N_stimuli, N_channels, N_timepoints)
        Y_train_3d = f[f"train/neural_data/{target_key}"][:]
        Y_test_3d  = f[f"test/neural_data/{target_key}"][:]
        nc_2d      = f[f"noise_ceilings/{target_key}"][:]

    if n_train: Y_train_3d = Y_train_3d[:n_train]
    if n_test:  Y_test_3d  = Y_test_3d[:n_test]
    
    # Store dimensions for reshaping later
    N_C, N_T = Y_train_3d.shape[1], Y_train_3d.shape[2]

    # Flatten for modeling: (N_stimuli, C * T)
    Y_train = Y_train_3d.reshape(Y_train_3d.shape[0], N_C * N_T).astype(np.float32)
    Y_test  = Y_test_3d.reshape(Y_test_3d.shape[0], N_C * N_T).astype(np.float32)
    nc_flat = nc_2d.flatten()

    # ---------------------------------------------------------
    # 2. FEATURE INDEX MAPS
    # ---------------------------------------------------------
    feat_file = "things_stimuli.h5" 
    path_a = os.path.join(feat_dir, model_a, feat_file)
    path_b = os.path.join(feat_dir, model_b, feat_file)

    idx_train = get_feat_indices(path_a, stim_ids_train)
    idx_test  = get_feat_indices(path_a, stim_ids_test)
    
    results = []

    # ---------------------------------------------------------
    # 3. EVALUATION LOOP
    # ---------------------------------------------------------
    for model_name, feat_path in [("Model A (ResNet)", path_a), ("Model B (Qwen3)", path_b)]:
        layers = list_h5_layers(feat_path)
        if n_layers: layers = layers[:n_layers]
        
        print(f"\n  ▶ {model_name} ({len(layers)} layers)")
        model_start = time.time()

        for layer_i, layer in enumerate(layers):
            layer_start = time.time()

            X_train = h5_indexed_read(feat_path, layer, idx_train)
            X_test  = h5_indexed_read(feat_path, layer, idx_test)

            try:
                # Use CLOSED-FORM fit_ridge_encoding here
                beta, mean, std, y_mean, best_alpha = fit_ridge_encoding_closed_form(
                    X_train, Y_train, DEVICE, alpha_grid=alpha_grid
                )
                
                # Evaluate using our unified function
                metrics, Y_pred, per_channel = eval_encoding(
                    beta, mean, std, y_mean, X_test, Y_test, 
                    nc_flat, DEVICE, nc_scale=nc_scale, nc_threshold=nc_threshold
                )
                
                # Reshape metrics back to (Channels, Timepoints)
                r_2d     = per_channel["r"].reshape(N_C, N_T)
                ev_2d    = per_channel["ev"].reshape(N_C, N_T)
                r_nc_2d  = per_channel["r_nc"].reshape(N_C, N_T)
                ev_nc_2d = per_channel["ev_nc"].reshape(N_C, N_T)

                # RSA and CKA (Calculated on the flattened C*T arrays)
                try:    enc_rsa = rsa_fn(Y_pred, Y_test)
                except: enc_rsa = float("nan")
                
                try:    enc_cka = cka_fn(Y_pred, Y_test)
                except: enc_cka = float("nan")

                results.append({
                    "dataset":           "EEG2",
                    "model":             model_name,
                    "layer":             layer,
                    "target":            roi,
                    "alpha":             best_alpha,
                    **metrics,           # Scalar means over C*T
                    "r_per_ch_t":        r_2d,
                    "ev_per_ch_t":       ev_2d,
                    "r_nc_per_ch_t":     r_nc_2d,
                    "ev_nc_per_ch_t":    ev_nc_2d,
                    "encoding_rsa":      enc_rsa,
                    "encoding_cka":      enc_cka,
                })
                
                print(f"    [{layer_i+1:2d}/{len(layers)}] {layer.split('/')[-1]:<28} | r={metrics['pearsonr']:.3f} (Peak ~100ms: {r_2d[:, 10].mean():.3f}) | α={best_alpha:.0e} | {time.time()-layer_start:.1f}s")
                
            except Exception as e:
                print(f"    ⚠ Fit failed [{target_key}/{layer}]: {e}")

            torch.cuda.empty_cache()

        print(f"  ✓ {model_name} done in {time.time()-model_start:.1f}s")

    print(f"\n✅ Finished {target_key} in {time.time()-wall_start:.1f}s")
    return pd.DataFrame(results)