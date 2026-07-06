#!/usr/bin/env python
# prefetch_models.py
#
# Prepare the TRIDENT pipeline to run on an offline GPU node.

import os
import sys
import json
from pathlib import Path

# repo root is two levels up: scripts/feature_extraction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIDENT_ROOT = PROJECT_ROOT / "TRIDENT"

# Local pathology TRIDENT must win over the site-packages package of the same name.
sys.path.insert(0, str(TRIDENT_ROOT))

# Redirect downloads into the repo (must match extract_features_trident.py, and
# be set before importing trident/huggingface).
_CACHE_ROOT = PROJECT_ROOT / "model_cache"
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))

from huggingface_hub import hf_hub_download

# The exact checkpoint file each TRIDENT model loads via torch.load(), and the
# local_ckpts.json registry key TRIDENT looks the path up under.
UNI_REPO, UNI_CKPT, UNI_KEY = "MahmoodLab/UNI", "pytorch_model.bin", "uni_v1"
SEG_REPO, SEG_CKPT, SEG_KEY = "MahmoodLab/hest-tissue-seg", "deeplabv3_seg_v4.ckpt", "hest"


def register_ckpt(subpkg: str, key: str, path: str) -> None:
    """Point TRIDENT's local_ckpts.json at a downloaded checkpoint so the offline
    compute nodes load it directly and skip the internet check."""
    reg_file = TRIDENT_ROOT / "trident" / subpkg / "local_ckpts.json"
    reg = json.loads(reg_file.read_text())
    reg[key] = path
    reg_file.write_text(json.dumps(reg, indent=4))
    print(f"    registered {subpkg}/{key}")


def main() -> int:
    print(f"Prefetching model weights into {_CACHE_ROOT} (needs internet)...")

    print(f"\t-> UNI patch encoder ({UNI_REPO}, gated)...")
    uni_path = hf_hub_download(repo_id=UNI_REPO, filename=UNI_CKPT)
    register_ckpt("patch_encoder_models", UNI_KEY, uni_path)

    print(f"\t-> hest segmentation checkpoint ({SEG_REPO})...")
    seg_path = hf_hub_download(repo_id=SEG_REPO, filename=SEG_CKPT)
    register_ckpt("segmentation_models", SEG_KEY, seg_path)

    # Load both models once (still online here). This caches the ResNet-50
    # backbone into TORCH_HOME and validates that the registered local paths
    # actually load, what the offline nodes will do.
    print("\t-> validating offline load (caches ResNet-50 backbone)...")
    from trident.segmentation_models import segmentation_model_factory
    from trident.patch_encoder_models import encoder_factory
    segmentation_model_factory("hest")
    encoder_factory(UNI_KEY)

    print(f"\nDone. Weights cached under {_CACHE_ROOT} and registered in "
          f"TRIDENT/local_ckpts.json. The offline GPU jobs can now load them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
