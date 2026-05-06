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
    print(f"\n{'='*60}\n🚀 STARTING PIPELINE: {target_key}\n{'='*60}")
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

# def run_tvsd_pipeline (roi: str,
#     monkey: str = "monkeyF",
#     data_dir: str = "/shared/NX-414/data/",
#     feat_dir: str = "/shared/NX-414/extracted_features/",
#     model_a: str = "adv_resnet152_imagenet_full_ffgsm_eps-1_alpha-125-ep10_seed-0",
#     model_b: str = "Qwen3-VL-2B-Instruct",
#     n_train: int = None,
#     n_test: int = None,
#     n_layers: int = None,
#     alpha_grid: np.ndarray = None,
#     nc_scale: float = 100.0,
#     nc_threshold: float = 10.0
# ) -> pd.DataFrame:

#     """
#     Runs the full linear encoding and RSA/CKA pipeline for a specific TVSD ROI.
#     """
#     if alpha_grid is None:
#         alpha_grid = np.logspace(0, 7, 15)

#     target_key = f"{monkey}/{roi}"
#     print(f"\n{'='*60}\n🚀 STARTING PIPELINE: {target_key}\n{'='*60}")
#     wall_start = time.time()

#     # ---------------------------------------------------------
#     # 1. LOAD NEURAL DATA & NOISE CEILING
#     # ---------------------------------------------------------
#     tvsd_path = os.path.join(data_dir, "tvsd.h5")
#     with h5py.File(tvsd_path, "r") as f:
#         stim_ids_train = f["train/stimulus_ids"][:]
#         stim_ids_test  = f["test/stimulus_ids"][:]
        
#         # Apply truncation if specified (for testing)
#         if n_train: stim_ids_train = stim_ids_train[:n_train]
#         if n_test:  stim_ids_test  = stim_ids_test[:n_test]
        
#         Y_train = f[f"train/neural_data/{target_key}"][:]
#         Y_test  = f[f"test/neural_data/{target_key}"][:]
#         nc_raw  = f[f"noise_ceilings/{target_key}"][:]

#     if n_train: Y_train = Y_train[:n_train]
#     if n_test:  Y_test  = Y_test[:n_test]
    
#     nc_flat = nc_raw.flatten() if nc_raw.ndim > 1 else nc_raw

#     # ---------------------------------------------------------
#     # 2. FEATURE INDEX MAPS
#     # ---------------------------------------------------------
#     feat_file = "things_stimuli.h5" # Adjust if TVSD uses a different filename
#     path_a = os.path.join(feat_dir, model_a, feat_file)
#     path_b = os.path.join(feat_dir, model_b, feat_file)

#     idx_train = get_feat_indices(path_a, stim_ids_train)
#     idx_test  = get_feat_indices(path_a, stim_ids_test)
    
#     TEST_MODE = True

    
#     if TEST_MODE:
#         print("⏳ INTERMEDIATE MODE — TVSD monkeyF/IT")
#         N_TRAIN, N_TEST = 2000, 200
#         N_LAYERS        = 3
#         ALPHA_GRID_RUN  = np.logspace(4, 6, 3)
    
#         tvsd_neural_run = {target_key: {
#             "train": tvsd_neural[target_key]["train"][:N_TRAIN],
#             "test":  tvsd_neural[target_key]["test"][:N_TEST],
#         }}
#         idx_tvsd_train_run = feat_idx_tvsd_train[:N_TRAIN]
#         idx_tvsd_test_run  = feat_idx_tvsd_test[:N_TEST]
#     else:
#         print("🚀 FULL RUN — TVSD monkeyF/IT")
#         N_LAYERS       = None
#         ALPHA_GRID_RUN = np.logspace(5, 5, 1)
    
#         tvsd_neural_run    = tvsd_neural          # already {TVSD_TARGET: {...}}
#         idx_tvsd_train_run = feat_idx_tvsd_train
#         idx_tvsd_test_run  = feat_idx_tvsd_test
    
#     tvsd_nc_run = tvsd_nc                          # already {TVSD_TARGET: nc_flat}
#     # ─────────────────────────────────────────────────────────────
#     # CELL 2.4.5  ·  Main evaluation loop (EEG only)
#     # ─────────────────────────────────────────────────────────────
        
#     def _fmt_time(seconds: float) -> str:
#         m, s = divmod(int(seconds), 60)
#         return f"{m}m {s:02d}s"
    
#     results = []
    
#     datasets_cfg = [
#         ("TVSD",
#          path_a_tvsd, path_b_tvsd,
#          idx_tvsd_train_run, idx_tvsd_test_run,
#          tvsd_neural_run, tvsd_nc_run),
#     ]
    
#     total_jobs = len(list_h5_layers(path_a_tvsd)) * len(tvsd_neural_run) * 2
#     print(f"Total jobs to run: {total_jobs}  (TVSD × models × layers × targets)\n")
    
#     rsa_fn = RepresentationalSimilarityAnalysis(dissimilarity="correlation",
#                                                  similarity_metric="spearman")
#     cka_fn = CenteredKernelAlignment()
    
#     job_done   = 0
#     wall_start = time.time()
    
#     for (ds_name,
#          feat_path_A, feat_path_B,
#          idx_train, idx_test,
#          neural_dict, nc_dict) in datasets_cfg:
    
#         print(f"\n{'='*60}")
#         print(f"  Dataset : {ds_name}")
#         print(f"  Targets : {list(neural_dict.keys())}")
#         ds_start = time.time()
#         print(f"{'='*60}")
    
#         for model_name, feat_path_src in [
#             ("Model A (ResNet)", feat_path_A),
#             ("Model B (Qwen3)",  feat_path_B),
#         ]:
#             layers = list_h5_layers(feat_path_src)
#             if N_LAYERS:
#                 layers = layers[:N_LAYERS]
#             n_layers = len(layers)
#             print(f"\n  ▶ {model_name}  ({n_layers} layers)")
#             model_start = time.time()
    
#             for layer_i, layer in enumerate(layers):
#                 layer_start = time.time()
    
#                 X_train = h5_indexed_read(feat_path_src, layer, idx_train)
#                 X_test  = h5_indexed_read(feat_path_src, layer, idx_test)
    
#                 target_scores = []
#                 for target_key, neural_splits in neural_dict.items():
#                     Y_train = neural_splits["train"].astype(np.float32)
#                     Y_test  = neural_splits["test"].astype(np.float32)
#                     nc_flat = nc_dict[target_key]
#                     if nc_flat.ndim > 1:
#                         nc_flat = nc_flat.flatten()
    
#                     try:
#                         t0 = time.time()
#                         beta, mean, std, y_mean, best_alpha = fit_ridge_encoding_adam(X_train, Y_train, alpha_grid=ALPHA_GRID_RUN)
#                         fit_t = time.time() - t0
#                     except Exception as e:
#                         print(f"    ⚠ Fit failed [{ds_name}/{target_key}/{layer}]: {e}")
#                         job_done += 1
#                         continue
                    
    
#                     metrics, Y_pred, per_channel = eval_encoding(beta, mean, std, y_mean, X_test, Y_test, nc_flat)
                    
#                     # ─── DIAGNOSTIC ───
#                     print(f"        Y_test  mean={Y_test.mean():.3f}  std={Y_test.std():.3f}")
#                     print(f"        Y_pred  mean={Y_pred.mean():.3f}  std={Y_pred.std():.3f}")
#                     print(f"        Y_train mean={Y_train.mean():.3f}  std={Y_train.std():.3f}")
#                     print(f"        Pearson r (mean over channels): {pearson_r_per_unit(Y_test, Y_pred).mean():.3f}")
#                     print(f"        EV       (mean over channels): {explained_variance_per_unit(Y_test, Y_pred).mean():.3f}")
#                     # ─────────────────
                    
                                    
#                     try:
#                         enc_rsa = rsa_fn(Y_pred, Y_test)
#                     except Exception:
#                         enc_rsa = float("nan")
    
#                     try:
#                         enc_cka = cka_fn(Y_pred, Y_test)
#                     except Exception:
#                         enc_cka = float("nan")
    
#                     results.append({
#                         "dataset":           ds_name,
#                         "model":             model_name,
#                         "layer":             layer,
#                         "target":            target_key,
#                         "alpha":             best_alpha,
#                         **metrics,
#                         "r_per_channel":     per_channel["r"],
#                         "ev_per_channel":    per_channel["ev"],
#                         "r_nc_per_channel":  per_channel["r_nc"],
#                         "ev_nc_per_channel": per_channel["ev_nc"],
#                         "encoding_rsa":      enc_rsa,
#                         "encoding_cka":      enc_cka,
#                     })
    
#                     target_scores.append(
#                         f"{target_key.split('/')[-1]}  "
#                         f"r={metrics['pearsonr']:.3f}  "
#                         f"r_nc={metrics['pearsonr_nc']:.3f}  "
#                         f"α={best_alpha:.0e}  "
#                         f"fit={fit_t:.1f}s"
#                     )
#                     job_done += 1
    
#                 layer_elapsed = time.time() - layer_start
#                 elapsed_total = time.time() - wall_start
#                 eta = (elapsed_total / max(job_done, 1)) * (total_jobs - job_done)
    
#                 layer_short = layer.split("/")[-1]
#                 scores_str  = " | ".join(target_scores) if target_scores else "no targets"
#                 print(
#                     f"    [{layer_i+1:2d}/{n_layers}] {layer_short:<28} "
#                     f"{scores_str}   "
#                     f"(layer {layer_elapsed:.1f}s | "
#                     f"elapsed {_fmt_time(elapsed_total)} | "
#                     f"ETA {_fmt_time(eta)})"
#                 )
    
#             model_elapsed = time.time() - model_start
#             print(f"\n  ✓ {model_name} done in {_fmt_time(model_elapsed)}")
    
#         ds_elapsed = time.time() - ds_start
#         print(f"\n  ✓✓ {ds_name} complete in {_fmt_time(ds_elapsed)}")
    
#     total_elapsed = time.time() - wall_start
#     df_results = pd.DataFrame(results)
#     print(f"\n{'='*60}")
#     print(f"✅  All done in {_fmt_time(total_elapsed)} — "
#           f"{len(df_results)} rows in results table.")
#     print(f"{'='*60}")
#     df_results.head()