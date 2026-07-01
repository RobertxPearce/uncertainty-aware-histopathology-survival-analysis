import os
import glob
import h5py
import torch
import torch.nn as nn
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path

from trident import OpenSlideWSI, visualize_heatmap
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory
from trident.slide_encoder_models import ABMILSlideEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================
# 0. Logging Configuration
# ==========================================
LOG_DIR = str(PROJECT_ROOT / "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("trident_run_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# ABMIL Model Definition (For Heatmap Visualization)
# ==========================================
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=1024, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim,
            n_heads=1,
            head_dim=512,
            dropout=0.0,
            gated=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)
        if return_raw_attention:
            return logits, attn
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N slides (smoke test).")
    args = parser.parse_args()

    logger.info("Trident Batch Feature Extraction & Heatmap Pipeline Started!")

    # ==========================================
    # 1. Environment and Path Setup
    # ==========================================
    WSI_DIR = str(PROJECT_ROOT / "data" / "raw" / "slides")
    OUTPUT_DIR = str(PROJECT_ROOT / "data" / "processed" / "trident_full")
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features_uni_v1")
    HEATMAP_OUT_DIR = os.path.join(OUTPUT_DIR, "heatmap_viz")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(HEATMAP_OUT_DIR, exist_ok=True)

    TARGET_MAG = 20
    PATCH_SIZE = 256
    PATCH_ENCODER = "uni_v1"

    device_str = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Slides are nested as data/raw/slides/<file_id>/<file_name>.svs
    svs_list = sorted(glob.glob(os.path.join(WSI_DIR, "**", "*.svs"), recursive=True))
    logger.info(f"Found a total of {len(svs_list)} slides.")
    if args.limit is not None:
        svs_list = svs_list[:args.limit]
        logger.info(f"--limit set: processing only {len(svs_list)} slide(s).")

    if not svs_list:
        logger.warning(f"Warning: No .svs files found in {WSI_DIR}. Terminating script.")
        return

    # ==========================================
    # STEP 1: Feature Extraction
    # ==========================================
    logger.info("=== [STEP 1] Loading Segmentation & Feature Extraction Models ===")
    try:
        segmentation_model = segmentation_model_factory("hest")
        patch_encoder = encoder_factory(PATCH_ENCODER).eval().to(device)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return

    total_slides = len(svs_list)

    for i, svs_path in enumerate(svs_list, 1):
        filename = os.path.basename(svs_path)
        h5_filename = filename.replace('.svs', '.h5')
        h5_path = os.path.join(FEATURES_DIR, h5_filename)

        logger.info(f"[{i}/{total_slides}] Feature Extraction: {filename}")

        if os.path.exists(h5_path):
            logger.info("  -> Feature file (.h5) already exists. Skipping.")
            continue

        start_time = time.time()
        try:
            slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False)

            logger.info("  -> 1/3. Segmentation started...")
            slide.segment_tissue(segmentation_model=segmentation_model, target_mag=10, job_dir=OUTPUT_DIR, device=device_str)

            logger.info("  -> 2/3. Patch coordinate extraction started...")
            coords_path = slide.extract_tissue_coords(target_mag=TARGET_MAG, patch_size=PATCH_SIZE, save_coords=OUTPUT_DIR)

            logger.info("  -> 3/3. Feature extraction (UNI) started... (Saving .h5)")
            slide.extract_patch_features(
                patch_encoder=patch_encoder,
                coords_path=coords_path,
                save_features=FEATURES_DIR,
                device=device_str
            )

            elapsed_time = time.time() - start_time
            logger.info(f"  -> Success! (Time elapsed: {elapsed_time:.1f}s)")

        except Exception as e:
            logger.error(f"  -> Failed! Error log: {str(e)}")

    # Clear heavy models from memory before heatmap generation
    del segmentation_model
    del patch_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("=== STEP 1 Feature Extraction Completed ===")

    # ==========================================
    # STEP 2: Heatmap Generation
    # ==========================================
    logger.info("=== [STEP 2] Loading ABMIL Heatmap Model ===")
    abmil_model = BinaryClassificationModel(input_feature_dim=1024).eval().to(device)

    heatmap_success_count = 0

    for i, svs_path in enumerate(svs_list, 1):
        filename = os.path.basename(svs_path)
        h5_filename = filename.replace('.svs', '.h5')
        h5_path = os.path.join(FEATURES_DIR, h5_filename)

        logger.info(f"[{i}/{total_slides}] Heatmap Generation: {filename}")

        if not os.path.exists(h5_path):
            logger.warning("  -> .h5 feature file not found. Skipping heatmap.")
            continue

        slide_heatmap_dir = os.path.join(HEATMAP_OUT_DIR, filename.replace('.svs', ''))
        if os.path.exists(slide_heatmap_dir) and os.listdir(slide_heatmap_dir):
            logger.info("  -> Heatmap already exists. Skipping.")
            heatmap_success_count += 1
            continue

        try:
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords'][:]
                patch_features = f['features'][:]
                coords_attrs = dict(f['coords'].attrs)

            batch = {'features': torch.from_numpy(patch_features).float().to(device).unsqueeze(0)}
            with torch.no_grad():
                _, attention = abmil_model(batch, return_raw_attention=True)

            slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False)

            heatmap_save_path = visualize_heatmap(
                wsi=slide,
                scores=attention.detach().cpu().numpy().squeeze(),
                coords=coords,
                vis_level=1,
                patch_size_level0=coords_attrs.get('patch_size_level0', 256),
                normalize=True,
                num_top_patches_to_save=0,
                output_dir=slide_heatmap_dir
            )
            logger.info("  -> Heatmap saved successfully!")
            heatmap_success_count += 1

        except Exception as e:
            logger.error(f"  -> Failed to generate heatmap! Error log: {str(e)}")

    logger.info("=" * 50)
    logger.info(f"All processing completed! (Heatmaps Generated: {heatmap_success_count}/{total_slides})")
    logger.info(f"Log file saved to: {log_filepath}")


if __name__ == "__main__":
    main()
