"""
Deployment Memory Profiler for AI Brain Tumor Analysis.
Measures process RSS memory and PyTorch memory at each step of startup and inference.
"""

import os
import sys
import gc
from pathlib import Path
import psutil

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_process_memory_mb():
    """Returns current process Resident Set Size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def profile():
    print("=" * 70)
    print("AI BRAIN TUMOR ANALYSIS - DEPLOYMENT MEMORY PROFILING")
    print("=" * 70)

    # A. Python process before importing torch/models
    mem_base = get_process_memory_mb()
    print(f"[A] Baseline Process Memory (before imports): {mem_base:.2f} MB")

    # Import torch
    import torch
    mem_torch = get_process_memory_mb()
    print(f"    After importing torch:                    {mem_torch:.2f} MB (+{mem_torch - mem_base:.2f} MB)")

    # Import models
    from src.models import build_resnet18_classifier
    from src.segmentation_model import UNet
    from src.explainability import GradCAM
    from backend.inference import run_full_inference
    mem_imported = get_process_memory_mb()
    print(f"    After importing application code:         {mem_imported:.2f} MB (+{mem_imported - mem_torch:.2f} MB)")

    # B. Inspect classifier checkpoint structure & memory
    cls_path = PROJECT_ROOT / "outputs" / "best_classifier.pth"
    print("\n--- Inspecting Classifier Checkpoint ---")
    mem_before_cls = get_process_memory_mb()
    cls_raw = torch.load(cls_path, map_location="cpu", weights_only=False)
    mem_cls_loaded_raw = get_process_memory_mb()
    print(f"    Memory with raw classifier checkpoint in RAM: {mem_cls_loaded_raw:.2f} MB (+{mem_cls_loaded_raw - mem_before_cls:.2f} MB)")
    print(f"    Keys in classifier checkpoint: {list(cls_raw.keys())}")
    
    # Check size of optimizer state dict vs model state dict
    import sys
    model_sd = cls_raw.get("model_state_dict", {})
    optim_sd = cls_raw.get("optimizer_state_dict", {})
    print(f"    Number of tensor entries in model_state_dict:     {len(model_sd)}")
    print(f"    Optimizer state dict present:                     {bool(optim_sd)}")

    classifier = build_resnet18_classifier(num_classes=4, pretrained=False, freeze_backbone=False)
    classifier.load_state_dict(model_sd)
    classifier.eval()
    mem_cls_instantiated = get_process_memory_mb()
    print(f"[B] Classifier instantiated & loaded in model:    {mem_cls_instantiated:.2f} MB (+{mem_cls_instantiated - mem_before_cls:.2f} MB)")

    del cls_raw, model_sd, optim_sd
    gc.collect()
    mem_after_cls_gc = get_process_memory_mb()
    print(f"    After deleting raw checkpoint & gc.collect():     {mem_after_cls_gc:.2f} MB")

    # Grad-CAM init
    gradcam = GradCAM(model=classifier, target_layer=classifier.layer4)
    mem_after_gradcam_init = get_process_memory_mb()
    print(f"    After Grad-CAM initialized on layer4:             {mem_after_gradcam_init:.2f} MB")

    # C. Inspect U-Net checkpoint structure & memory
    unet_path = PROJECT_ROOT / "outputs" / "best_unet.pth"
    print("\n--- Inspecting U-Net Checkpoint ---")
    mem_before_unet = get_process_memory_mb()
    unet_raw = torch.load(unet_path, map_location="cpu", weights_only=False)
    mem_unet_loaded_raw = get_process_memory_mb()
    print(f"    PEAK Memory with raw U-Net checkpoint in RAM:    {mem_unet_loaded_raw:.2f} MB (+{mem_unet_loaded_raw - mem_before_unet:.2f} MB)")
    print(f"    Keys in U-Net checkpoint: {list(unet_raw.keys())}")
    
    unet_sd = unet_raw.get("model_state_dict", {})
    unet_optim_sd = unet_raw.get("optimizer_state_dict", {})
    print(f"    Number of tensor entries in U-Net model_state_dict: {len(unet_sd)}")
    print(f"    Optimizer state dict present in U-Net checkpoint:   {bool(unet_optim_sd)}")

    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(unet_sd)
    unet.eval()
    mem_unet_instantiated = get_process_memory_mb()
    print(f"[C] U-Net instantiated & loaded in model:          {mem_unet_instantiated:.2f} MB (+{mem_unet_instantiated - mem_before_unet:.2f} MB)")

    del unet_raw, unet_sd, unet_optim_sd
    gc.collect()
    mem_after_unet_gc = get_process_memory_mb()
    print(f"[D] After deleting raw U-Net checkpoint & gc.collect(): {mem_after_unet_gc:.2f} MB")

    # E, F, G, H: Inference memory profiling
    print("\n--- Inference Memory Profiling ---")
    sample_path = PROJECT_ROOT / "brisc2025" / "classification_task" / "test" / "glioma" / "brisc2025_test_00001_gl_ax_t1.jpg"
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    mem_before_infer = get_process_memory_mb()
    print(f"    Memory before inference call:                     {mem_before_infer:.2f} MB")

    # Run inference via backend
    result = run_full_inference(img_bytes)
    mem_during_infer = get_process_memory_mb()
    print(f"[E, F, G] Memory after run_full_inference():          {mem_during_infer:.2f} MB (+{mem_during_infer - mem_before_infer:.2f} MB)")
    print(f"    Result prediction: {result['prediction']['class']} ({result['prediction']['confidence_percentage']}%)")

    # Cleanup
    del result, img_bytes
    gc.collect()
    mem_after_cleanup = get_process_memory_mb()
    print(f"[H] Memory after gc.collect():                        {mem_after_cleanup:.2f} MB")

    print("\n" + "=" * 70)
    print("SUMMARY OF RAM FINDINGS:")
    print(f"  Baseline Python + PyTorch:               {mem_imported:.2f} MB")
    print(f"  PEAK RAM During Unoptimized U-Net Load:  {mem_unet_loaded_raw:.2f} MB")
    print(f"  Settled RAM With Both Models In Memory:  {mem_after_unet_gc:.2f} MB")
    print(f"  Settled RAM After Full Inference:        {mem_after_cleanup:.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    profile()
