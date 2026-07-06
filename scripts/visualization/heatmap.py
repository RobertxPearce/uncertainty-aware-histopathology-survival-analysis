import os
import glob
import h5py
import torch
import torch.nn as nn
import logging
import time
from datetime import datetime

from trident import OpenSlideWSI, visualize_heatmap
from trident.slide_encoder_models import ABMILSlideEncoder

# ==========================================
# 0. Logging Configuration
# ==========================================
LOG_DIR = '/Users/sejun/Downloads/Trident_Logs'
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("heatmap_gen_%Y%m%d_%H%M%S.log")
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
    logger.info("ABMIL Heatmap Generation Pipeline Started!")

    # ==========================================
    # 1. Environment and Path Setup
    # ==========================================
    WSI_DIR = '/Users/sejun/Downloads/pilot100_slides'
    OUTPUT_DIR = '/Users/sejun/Downloads/Trident_Batch_Output'
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features_uni_v1")
    HEATMAP_OUT_DIR = os.path.join(OUTPUT_DIR, "heatmap_viz")

    os.makedirs(HEATMAP_OUT_DIR, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    svs_list = glob.glob(os.path.join(WSI_DIR, "*.svs"))
    logger.info(f"Found a total of {len(svs_list)} slides.")

    if not svs_list:
        logger.warning(f"Warning: No .svs files found in {WSI_DIR}. Terminating script.")
        return

    # ==========================================
    # 2. Loading ABMIL Heatmap Model
    # ==========================================
    logger.info("Loading ABMIL Heatmap Model...")
    try:
        abmil_model = BinaryClassificationModel(input_feature_dim=1024).eval().to(device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # ==========================================
    # 3. Heatmap Generation
    # ==========================================
    total_slides = len(svs_list)
    success_count = 0

    for i, svs_path in enumerate(svs_list, 1):
        filename = os.path.basename(svs_path)
        h5_filename = filename.replace('.svs', '.h5')
        h5_path = os.path.join(FEATURES_DIR, h5_filename)
        
        logger.info(f"[{i}/{total_slides}] Heatmap Generation: {filename}")
        
        if not os.path.exists(h5_path):
            logger.warning(f"  -> .h5 feature file not found ({h5_filename}). Skipping heatmap.")
            continue
            
        try:
            # Load features and coordinates
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords'][:]
                patch_features = f['features'][:]
                coords_attrs = dict(f['coords'].attrs)
                
            # Get attention scores
            batch = {'features': torch.from_numpy(patch_features).float().to(device).unsqueeze(0)}
            with torch.no_grad():
                _, attention = abmil_model(batch, return_raw_attention=True)
                
            # Initialize slide and generate heatmap
            slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False)
            
            # Create a unique output directory for each slide to prevent overwriting
            slide_heatmap_dir = os.path.join(HEATMAP_OUT_DIR, filename.replace('.svs', ''))
            
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
            success_count += 1
            
        except Exception as e:
            logger.error(f"  -> Failed to generate heatmap! Error log: {str(e)}")

    logger.info("="*50)
    logger.info(f"Heatmap generation completed! (Success: {success_count}/{total_slides})")
    logger.info(f"Log file saved to: {log_filepath}")

if __name__ == "__main__":
    main()
