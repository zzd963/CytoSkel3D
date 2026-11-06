"""Evaluate skeleton extraction on one set of paired TIFF images.

Run from the repository root, for example:
    python Demo/evaluate_skeleton_performance_3D.py Data/synthetic/reference Data/synthetic/predictions results/skeleton_evaluation

The example directories are placeholders. Supply reference files named
synth_0000_base.tiff and predictions arranged as
synth_0000/necessities/synth_0000_pre_cleaned_skeleton.tif.
Pairs must be on the same pixel or voxel grid. The default tolerance is
one pixel in 2D or one voxel in 3D.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.morphology import ball, dilation, disk


def calculate_metrics_with_tolerance(ground_truth, prediction, tolerance=1):
    """Return TPR, PPV, and DSC for two skeleton masks on the same grid."""
    if ground_truth.shape != prediction.shape or ground_truth.ndim not in (2, 3):
        raise ValueError("Skeletons must have the same shape and be 2D or 3D.")
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative.")

    # Positive pixels or voxels belong to the skeleton.
    gt_bool = ground_truth > 0
    pred_bool = prediction > 0

    # The tolerance neighborhood is a disk in 2D and a ball in 3D.
    footprint = ball(tolerance) if gt_bool.ndim == 3 else disk(tolerance)
    gt_tolerance = dilation(gt_bool, footprint)
    pred_tolerance = dilation(pred_bool, footprint)

    # TP and FP count prediction positions inside and outside the reference
    # tolerance region. FN counts reference positions outside the prediction
    # tolerance region.
    tp = np.sum(np.logical_and(pred_bool, gt_tolerance))
    fp = np.sum(np.logical_and(pred_bool, np.logical_not(gt_tolerance)))
    fn = np.sum(np.logical_and(gt_bool, np.logical_not(pred_tolerance)))

    # An undefined score with a zero denominator is reported as zero.
    tpr = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    dsc = (2 * tp) / (2 * tp + fp + fn) if tp else 0.0
    return tpr, ppv, dsc


def evaluate_skeleton_performance(ground_truth_dir, prediction_dir, tolerance=1):
    """Evaluate every reference image and append the mean across images."""
    ground_truth_dir = Path(ground_truth_dir)
    prediction_dir = Path(prediction_dir)
    reference_paths = sorted(ground_truth_dir.glob("synth_*_base.tiff"))
    if not reference_paths:
        raise FileNotFoundError(f"No synth_*_base.tiff files found in {ground_truth_dir}")

    results = []
    for reference_path in reference_paths:
        # Match the prediction to the reference by synthetic image ID.
        image_id = reference_path.stem.rsplit("_", 1)[0]
        prediction_path = (
            prediction_dir / image_id / "necessities" / f"{image_id}_pre_cleaned_skeleton.tif"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Prediction for {image_id} not found: {prediction_path}")

        ground_truth = tifffile.imread(reference_path)
        prediction = tifffile.imread(prediction_path)
        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"Skeleton grids differ for {image_id}: "
                f"reference {ground_truth.shape}, prediction {prediction.shape}."
            )

        tpr, ppv, dsc = calculate_metrics_with_tolerance(ground_truth, prediction, tolerance)
        results.append({"image_id": image_id, "TPR": tpr, "PPV": ppv, "DSC": dsc})

    scores = pd.DataFrame(results)
    # Average the per-image scores so each image has the same weight.
    means = scores[["TPR", "PPV", "DSC"]].mean().to_dict()
    scores.loc[len(scores)] = {"image_id": "Average", **means}
    return scores


def visualize_results(scores, output_dir):
    """Save the per-image metric plots and the distribution boxplot."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    # The last row contains summary values and is not an individual image.
    per_image = scores.iloc[:-1]
    means = scores.iloc[-1]
    metrics = ("TPR", "PPV", "DSC")
    titles = ("Sensitivity (TPR)", "Precision (PPV)", "Dice Similarity Coefficient (DSC)")

    figure, axes = plt.subplots(3, 1, figsize=(15, 10))
    for axis, metric, title, color in zip(axes, metrics, titles, ("blue", "green", "magenta")):
        axis.plot(per_image.index, per_image[metric], "o-", color=color, label=f"{metric} per image")
        axis.axhline(means[metric], color="red", linestyle="--", label=f"Mean {metric}: {means[metric]:.4f}")
        axis.set(title=title, ylabel=metric, ylim=(0, 1.1))
        axis.legend(loc="lower left")
    axes[-1].set_xlabel("Image index")
    figure.tight_layout()
    figure.savefig(output_dir / "skeleton_performance_metrics.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    values = per_image[list(metrics)].to_numpy()
    axis.boxplot(values, labels=metrics)
    axis.set(
        title="Distribution of Skeleton Extraction Performance Metrics",
        ylabel="Value",
        ylim=(max(0, values.min() - 0.1), 1.1),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "performance_boxplot.png")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one set of reference and predicted skeleton TIFFs.",
        epilog="Relative paths are resolved from the current working directory.",
    )
    parser.add_argument("ground_truth_dir", type=Path, help="Directory containing synth_*_base.tiff")
    parser.add_argument("prediction_dir", type=Path, help="Directory containing the per-image prediction folders")
    parser.add_argument("output_dir", type=Path, help="Directory for the CSV and metric plots")
    parser.add_argument("--tolerance", type=int, default=1, help="Tolerance in pixels or voxels (default: 1)")
    args = parser.parse_args()

    try:
        scores = evaluate_skeleton_performance(args.ground_truth_dir, args.prediction_dir, args.tolerance)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "performance_after_postprocess.csv"
    scores.to_csv(output_csv, index=False)
    visualize_results(scores, args.output_dir / "after_postprocess")
    print(f"Evaluated {len(scores) - 1} image pairs. Results saved to {output_csv}.")


if __name__ == "__main__":
    main()
