import os
import time
import h5py
import numpy as np
import pandas as pd
import torch

from metrics import *
from encoding import *
from data_io import *

def run_nsd_pipeline(
    subject: str,
    rois: list[str],    
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
    Runs the full linear encoding and RSA/CKA pipeline for a specific NSD Subject.
    Computes all requested ROIs simultaneously for maximum memory efficiency.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(0, 7, 15)

    print(f"\n{'='*60}\n STARTING NSD PIPELINE: {subject}\n  Targets: {rois}\n{'='*60}")
    wall_start = time.time()

    # ---------------------------------------------------------
    # 1. LOAD NEURAL DATA & NOISE CEILING FOR ALL ROIS
    # ---------------------------------------------------------
    nsd_path = os.path.join(data_dir, "nsd_func1pt8mm_individualROIs.h5")
    neural_dict = {}
    nc_dict = {}

    with h5py.File(nsd_path, "r") as f:
        stim_ids_train = f[f"train/stimulus_ids/{subject}"][:]
        stim_ids_test  = f[f"test/stimulus_ids/{subject}"][:]
        
        if n_train: stim_ids_train = stim_ids_train[:n_train]
        if n_test:  stim_ids_test  = stim_ids_test[:n_test]
        
        for roi in rois:
            Y_train = f[f"train/neural_data/{subject}/{roi}"][:]
            Y_test  = f[f"test/neural_data/{subject}/{roi}"][:]
            nc_raw  = f[f"noise_ceilings/{subject}/{roi}"][:].astype(np.float32)

            if n_train: Y_train = Y_train[:n_train]
            if n_test:  Y_test  = Y_test[:n_test]

            neural_dict[roi] = {
                "train": Y_train,
                "test":  Y_test
            }
            nc_dict[roi] = nc_raw.flatten() if nc_raw.ndim > 1 else nc_raw

    # ---------------------------------------------------------
    # 2. FEATURE INDEX MAPS
    # ---------------------------------------------------------
    feat_file = "nsd_stimuli.h5" 
    path_a = os.path.join(feat_dir, model_a, feat_file)
    path_b = os.path.join(feat_dir, model_b, feat_file)

    idx_train = get_feat_indices(path_a, stim_ids_train)
    idx_test  = get_feat_indices(path_a, stim_ids_test)
    
    results = []
    total_jobs = (len(list_h5_layers(path_a)) + len(list_h5_layers(path_b))) * len(rois)
    job_done = 0

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

            # Load heavy feature matrices ONCE per layer
            X_train = h5_indexed_read(feat_path, layer, idx_train)
            X_test  = h5_indexed_read(feat_path, layer, idx_test)

            target_scores = []
            
            # Loop through all ROIs using the same X_train memory block
            for roi_target, neural_splits in neural_dict.items():
                Y_train = neural_splits["train"].astype(np.float32)
                Y_test  = neural_splits["test"].astype(np.float32)
                nc_flat = nc_dict[roi_target]

                try:
                    # CLOSED-FORM fit_ridge_encoding
                    beta, mean, std, y_mean, best_alpha = fit_ridge_encoding_closed_form(
                        X_train, Y_train, DEVICE, alpha_grid=alpha_grid
                    )
                    
                    metrics, Y_pred, per_channel = eval_encoding(
                        beta, mean, std, y_mean, X_test, Y_test, 
                        nc_flat, DEVICE, nc_scale=nc_scale, nc_threshold=nc_threshold
                    )

                    try:    enc_rsa = rsa_fn(Y_pred, Y_test)
                    except: enc_rsa = float("nan")
                    
                    try:    enc_cka = cka_fn(Y_pred, Y_test)
                    except: enc_cka = float("nan")

                    results.append({
                        "dataset":           "FMRI",
                        "model":             model_name,
                        "layer":             layer,
                        "target":            roi_target, 
                        "alpha":             best_alpha,
                        **metrics,
                        "r_per_channel":     per_channel["r"],
                        "ev_per_channel":    per_channel["ev"],
                        "r_nc_per_channel":  per_channel["r_nc"],
                        "ev_nc_per_channel": per_channel["ev_nc"],
                        "encoding_rsa":      enc_rsa,
                        "encoding_cka":      enc_cka,
                    })
                    
                    target_scores.append(f"{roi_target}: r={metrics['pearsonr']:.3f}/r_nc={metrics['pearsonr_nc']:.3f}")
                    job_done += 1
                    
                except Exception as e:
                    print(f"    ⚠ Fit failed [{subject}/{roi_target}/{layer}]: {e}")
                    job_done += 1

                # Clear VRAM after each ROI
                del beta, Y_pred
                torch.cuda.empty_cache()

            # Beautiful summary printout restored
            layer_elapsed = time.time() - layer_start
            scores_str = "  |  ".join(target_scores) if target_scores else "no targets"
            layer_short = layer.split("/")[-1]
            print(f"    [{layer_i+1:2d}/{len(layers)}] {layer_short:<28}\n        {scores_str}\n        (layer {layer_elapsed:.1f}s)")

        print(f"  ✓ {model_name} done in {time.time()-model_start:.1f}s")

    print(f"\n✅ Finished {subject} in {time.time()-wall_start:.1f}s")
    return pd.DataFrame(results)