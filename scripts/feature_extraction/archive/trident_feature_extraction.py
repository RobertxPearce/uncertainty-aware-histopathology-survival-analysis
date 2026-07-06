import os
import glob
import torch
import logging
import time
import shutil
from datetime import datetime

from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

# ==========================================
# 0. Logging Configuration
# ==========================================
LOG_DIR = '/Users/sejun/Downloads/Trident_Logs'
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("trident_extract_%Y%m%d_%H%M%S.log")
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

def main():
    logger.info("Trident Feature Extraction Pipeline Started! (Generating ONLY .geojson, .h5, and logs)")

    # ==========================================
    # 1. Environment and Path Setup
    # ==========================================
    WSI_DIR = '/Users/sejun/Downloads/pilot100_slides'
    OUTPUT_DIR = '/Users/sejun/Downloads/Trident_Batch_Output'
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features_uni_v1")

    os.makedirs(FEATURES_DIR, exist_ok=True)

    TARGET_MAG = 20
    PATCH_SIZE = 256
    PATCH_ENCODER = "uni_v1"

    device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    svs_list = glob.glob(os.path.join(WSI_DIR, "*.svs"))
    logger.info(f"Found a total of {len(svs_list)} slides.")

    if not svs_list:
        logger.warning(f"Warning: No .svs files found in {WSI_DIR}. Terminating script.")
        return

    # ==========================================
    # 2. Initialize Models
    # ==========================================
    logger.info("Loading Segmentation & Feature Extraction Models...")
    try:
        segmentation_model = segmentation_model_factory("hest")
        patch_encoder = encoder_factory(PATCH_ENCODER).eval().to(device)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    total_slides = len(svs_list)
    success_count = 0
    
    for i, svs_path in enumerate(svs_list, 1):
        filename = os.path.basename(svs_path)
        h5_filename = filename.replace('.svs', '.h5')
        h5_path = os.path.join(FEATURES_DIR, h5_filename)
        
        logger.info(f"[{i}/{total_slides}] Feature Extraction: {filename}")
        
        if os.path.exists(h5_path):
            logger.info("  -> Feature file (.h5) already exists. Skipping.")
            success_count += 1
            continue
            
        start_time = time.time()
        try:
            slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False)
            
            logger.info("  -> 1/3. Segmentation started...")
            slide.segment_tissue(segmentation_model=segmentation_model, target_mag=10, job_dir=OUTPUT_DIR, device=device)
            
            logger.info("  -> 2/3. Patch coordinate extraction started...")
            coords_path = slide.extract_tissue_coords(target_mag=TARGET_MAG, patch_size=PATCH_SIZE, save_coords=OUTPUT_DIR)
            
            logger.info("  -> 3/3. Feature extraction (UNI) started... (Saving .h5)")
            slide.extract_patch_features(
                patch_encoder=patch_encoder, 
                coords_path=coords_path, 
                save_features=FEATURES_DIR, 
                device=device
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"  -> Success! (Time elapsed: {elapsed_time:.1f}s)")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  -> Failed! Error log: {str(e)}")

    # ==========================================
    # 4. Cleanup Unwanted Visualization Files
    # ==========================================
    logger.info("Cleaning up unwanted image files (thumbnails, contours)...")
    thumbnails_dir = os.path.join(OUTPUT_DIR, "thumbnails")
    contours_dir = os.path.join(OUTPUT_DIR, "contours")
    
    if os.path.exists(thumbnails_dir):
        shutil.rmtree(thumbnails_dir)
    if os.path.exists(contours_dir):
        shutil.rmtree(contours_dir)

    logger.info("="*50)
    logger.info(f"Feature extraction completed! (Success: {success_count}/{total_slides})")
    logger.info(f"Log file saved to: {log_filepath}")

if __name__ == "__main__":
    main()
