"""Run reproducible inference on one CornSeg test image and visualize it."""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as standard_transforms

from crowd_datasets.CornSeg.CornSeg import CornSeg
from models import build_model
from util.misc import nested_tensor_from_tensor_list


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-min-size", type=int, default=384)
    parser.add_argument("--eval-max-size", type=int, default=768)
    return parser.parse_args()


def model_args() -> Namespace:
    return Namespace(
        backbone="vgg16_bn",
        row=2,
        line=2,
        use_bifpn=False,
        use_p3=False,
        use_prm=False,
        use_old_model=False,
        use_old_inc_model=False,
        use_structural_model=False,
        prm_d_model=64,
        prm_nhead=4,
        prm_num_layers=2,
        prm_dim_feedforward=128,
        point_loss_coef=0.005,
        eos_coef=0.5,
    )


def detect_architecture(args: Namespace, state_dict: dict[str, torch.Tensor]) -> None:
    args.use_prm = any(key.startswith("prm.") for key in state_dict)
    args.use_bifpn = any(key.startswith("fpn.P4_td_fusion.") for key in state_dict)
    args.use_p3 = any(key.startswith("p3_adapter.") for key in state_dict)


def restore_image(tensor: torch.Tensor) -> np.ndarray:
    restored = tensor.detach().cpu().clone()
    for channel, mean, std in zip(restored, MEAN, STD):
        channel.mul_(std).add_(mean)
    return (
        restored.clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)


def main() -> None:
    cli = parse_args()
    if not 0 <= cli.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    device = torch.device(f"cuda:{cli.gpu_id}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cli.weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    architecture = model_args()
    detect_architecture(architecture, state_dict)
    model = build_model(architecture, training=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    dataset = CornSeg(
        str(cli.data_root),
        transform=transform,
        train=False,
        list_file="corn_seg_test.list",
        eval_min_size=cli.eval_min_size,
        eval_max_size=cli.eval_max_size,
    )
    matches = [
        index
        for index, image_path in enumerate(dataset.img_list)
        if Path(image_path).name == cli.image_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one test image named {cli.image_name}, found {len(matches)}")

    image_tensor, target_list = dataset[matches[0]]
    target = target_list[0]
    samples = nested_tensor_from_tensor_list([image_tensor]).to(device)

    with torch.inference_mode():
        outputs = model(samples)
        scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
        keep = scores > cli.threshold
        pred_points = outputs["pred_points"][0, keep].detach().cpu().numpy()
        pred_scores = scores[keep].detach().cpu().numpy()

    gt_points = target["point"].detach().cpu().numpy()
    gt_count = int(len(gt_points))
    pred_count = int(len(pred_points))
    absolute_error = abs(pred_count - gt_count)
    image = restore_image(image_tensor)
    height, width = image.shape[:2]
    inside = (
        (pred_points[:, 0] >= 0)
        & (pred_points[:, 0] < width)
        & (pred_points[:, 1] >= 0)
        & (pred_points[:, 1] < height)
    ) if pred_count else np.zeros(0, dtype=bool)

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cli.image_name).stem
    figure_path = cli.output_dir / f"{stem}_inference_comparison.png"
    points_path = cli.output_dir / f"{stem}_prediction_points.csv"
    summary_path = cli.output_dir / f"{stem}_inference_summary.json"

    figure, axes = plt.subplots(1, 3, figsize=(12.8, 5.8), constrained_layout=True)
    panels = (
        ("Input image", False, False),
        (f"Ground Truth\nGT = {gt_count}", True, False),
        (f"BiFPN+PRM prediction\nPred = {pred_count}   AE = {absolute_error}", True, True),
    )
    for axis, (title, show_gt, show_pred) in zip(axes, panels):
        axis.imshow(image)
        if show_gt:
            axis.scatter(
                gt_points[:, 0], gt_points[:, 1], s=14,
                facecolors="none", edgecolors="#00ff55", linewidths=0.7,
                label="Ground truth", zorder=3,
            )
        if show_pred and pred_count:
            axis.scatter(
                pred_points[inside, 0], pred_points[inside, 1], s=8,
                c="#e60000", marker="o", linewidths=0,
                label="Prediction", zorder=4,
            )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.axis("off")

    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=2, frameon=False)
    figure.suptitle(
        f"Test-set inference: {cli.image_name}   |   threshold = {cli.threshold:.2f}",
        fontsize=13, fontweight="bold",
    )
    figure.savefig(figure_path, dpi=300, facecolor="white")
    plt.close(figure)

    with points_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "confidence", "inside_valid_image"))
        for point, score, valid in zip(pred_points, pred_scores, inside):
            writer.writerow((float(point[0]), float(point[1]), float(score), bool(valid)))

    summary = {
        "image": cli.image_name,
        "split": "test",
        "model": "P2PNet-BiFPN-PRM",
        "checkpoint": str(cli.weight_path.resolve()),
        "device": str(device),
        "threshold": cli.threshold,
        "resized_image_width": width,
        "resized_image_height": height,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "absolute_error": absolute_error,
        "predictions_outside_valid_image": int((~inside).sum()) if pred_count else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figure: {figure_path.resolve()}")
    print(f"Prediction points: {points_path.resolve()}")


if __name__ == "__main__":
    main()
