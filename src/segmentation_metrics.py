import numpy as np
import torch
from typing import Dict, List, Tuple, Union


def compute_sample_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """
    Computes all standard binary segmentation metrics for a single sample:
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU / Jaccard)
    - Precision
    - Recall (Sensitivity)
    - Specificity
    - Pixel Accuracy
    
    Args:
        pred_mask: Binary numpy array [H, W] (0 or 1)
        target_mask: Binary numpy array [H, W] (0 or 1)
    """
    pred_flat = pred_mask.flatten().astype(bool)
    target_flat = target_mask.flatten().astype(bool)

    tp = np.logical_and(pred_flat, target_flat).sum()
    fp = np.logical_and(pred_flat, np.logical_not(target_flat)).sum()
    fn = np.logical_and(np.logical_not(pred_flat), target_flat).sum()
    tn = np.logical_and(np.logical_not(pred_flat), np.logical_not(target_flat)).sum()

    total_pixels = len(pred_flat)

    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    pixel_accuracy = (tp + tn) / total_pixels

    target_tumor_pixels = int(target_flat.sum())
    target_tumor_pct = (target_tumor_pixels / total_pixels) * 100.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "pixel_accuracy": float(pixel_accuracy),
        "tumor_pixels": target_tumor_pixels,
        "tumor_area_pct": float(target_tumor_pct)
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregates a list of per-sample metrics into summary statistics:
    mean, std, median, min, max for each metric.
    """
    keys = ["dice", "iou", "precision", "recall", "specificity", "pixel_accuracy"]
    summary = {}
    
    for k in keys:
        vals = np.array([m[k] for m in metrics_list])
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }
    return summary


def compute_metrics_by_tumor_size(
    metrics_list: List[Dict[str, float]],
    small_threshold_pct: float = 0.71,
    large_threshold_pct: float = 2.14
) -> Dict[str, Dict[str, float]]:
    """
    Computes Dice & IoU stratified by tumor area categories based strictly on training percentiles:
    - Small: < 25th percentile (0.71% of slice)
    - Medium: 25th - 75th percentile (0.71% - 2.14% of slice)
    - Large: > 75th percentile (2.14% of slice)
    """
    small_cases = [m for m in metrics_list if m["tumor_area_pct"] < small_threshold_pct]
    med_cases = [m for m in metrics_list if small_threshold_pct <= m["tumor_area_pct"] <= large_threshold_pct]
    large_cases = [m for m in metrics_list if m["tumor_area_pct"] > large_threshold_pct]

    groups = {
        "small_tumors": small_cases,
        "medium_tumors": med_cases,
        "large_tumors": large_cases
    }

    result = {}
    for gname, glist in groups.items():
        if glist:
            dices = [m["dice"] for m in glist]
            ious = [m["iou"] for m in glist]
            result[gname] = {
                "count": len(glist),
                "dice_mean": float(np.mean(dices)),
                "dice_std": float(np.std(dices)),
                "dice_median": float(np.median(dices)),
                "iou_mean": float(np.mean(ious)),
                "iou_std": float(np.std(ious)),
                "iou_median": float(np.median(ious))
            }
        else:
            result[gname] = {"count": 0, "dice_mean": 0.0, "iou_mean": 0.0}

    return result
