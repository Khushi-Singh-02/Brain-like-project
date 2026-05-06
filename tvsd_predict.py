from metrics import *
from encoding import *
from data_io import *
import os
import time
import h5py
import numpy as np
import pandas as pd
import torch

def run_tvsd_pipeline(
    roi: str,
    rsa_fn,            
    cka_fn,     
    DEVICE,
    monkey: str = "monkeyF",
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
    Runs the full linear encoding and RSA/CKA pipeline for a specific TVSD ROI.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(0, 7, 15)

    target_key = f"{monkey}/{roi}"
    print(f"\n{'='*60}\n STARTING PIPELINE: {target_key}\n{'='*60}")
    wall_start = time.time()

    # ---------------------------------------------------------
    # 1. LOAD NEURAL DATA & NOISE CEILING
    # ---------------------------------------------------------
    tvsd_path = os.path.join(data_dir, "tvsd.h5")
    with h5py.File(tvsd_path, "r") as f:
        stim_ids_train = f["train/stimulus_ids"][:]
        stim_ids_test  = f["test/stimulus_ids"][:]
        
        # Apply truncation if specified (for testing)
        if n_train: stim_ids_train = stim_ids_train[:n_train]
        if n_test:  stim_ids_test  = stim_ids_test[:n_test]
        
        Y_train = f[f"train/neural_data/{target_key}"][:]
        Y_test  = f[f"test/neural_data/{target_key}"][:]
        nc_raw  = f[f"noise_ceilings/{target_key}"][:]

    if n_train: Y_train = Y_train[:n_train]
    if n_test:  Y_test  = Y_test[:n_test]
    
    nc_flat = nc_raw.flatten() if nc_raw.ndim > 1 else nc_raw

    # ---------------------------------------------------------
    # 2. FEATURE INDEX MAPS
    # ---------------------------------------------------------
    feat_file = "things_stimuli.h5" # Adjust if TVSD uses a different filename
    path_a = os.path.join(feat_dir, model_a, feat_file)
    path_b = os.path.join(feat_dir, model_b, feat_file)

    idx_train = get_feat_indices(path_a, stim_ids_train)
    idx_test  = get_feat_indices(path_a, stim_ids_test)
    
    results = []

    # ---------------------------------------------------------
    # 4. EVALUATION LOOP
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
                # Assuming fit_ridge_encoding and eval_encoding are imported/defined
                beta, mean, std, y_mean, best_alpha = fit_ridge_encoding_adam(
                    X_train, Y_train.astype(np.float32), alpha_grid=alpha_grid, DEVICE=DEVICE
                )
                
                metrics, Y_pred, per_channel = eval_encoding(
                    beta, mean, std, y_mean, X_test, Y_test.astype(np.float32), 
                    nc_flat, DEVICE, nc_scale=nc_scale, nc_threshold=nc_threshold
                )
                
                try:    enc_rsa = rsa_fn(Y_pred, Y_test.astype(np.float32))
                except: enc_rsa = float("nan")
                
                try:    enc_cka = cka_fn(Y_pred, Y_test.astype(np.float32))
                except: enc_cka = float("nan")

                results.append({
                    "dataset":           "TVSD",
                    "model":             model_name,
                    "layer":             layer,
                    "target":            target_key,
                    "alpha":             best_alpha,
                    **metrics,
                    "encoding_rsa":      enc_rsa,
                    "encoding_cka":      enc_cka,
                })
                
                print(f"    [{layer_i+1:2d}/{len(layers)}] {layer.split('/')[-1]:<28} | r={metrics['pearsonr']:.3f} | α={best_alpha:.0e} | {time.time()-layer_start:.1f}s")
                
            except Exception as e:
                print(f"    ⚠ Fit failed [{target_key}/{layer}]: {e}")

            # Cleanup VRAM
            torch.cuda.empty_cache()

        print(f"  ✓ {model_name} done in {time.time()-model_start:.1f}s")

    print(f"\n✅ Finished {target_key} in {time.time()-wall_start:.1f}s")
    return pd.DataFrame(results)
