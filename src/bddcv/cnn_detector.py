"""Small hand-written CNN object detectors for the course comparison."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.ops import batched_nms
from torchvision.transforms import functional as TF

from .constants import DET_CLASSES
from .prediction import write_coco_predictions

IMAGE_SIZE = (640, 360)  # width, height; same 16:9 geometry as BDD100K
GRID_SIZE = (12, 20)     # rows, columns
BOXES_PER_CELL = 5


class ConvBlock(nn.Sequential):
    """Convolution, normalization, activation and 2x downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class ResidualBlock(nn.Module):
    """Two-convolution residual block used only by the complex model."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity() if stride == 1 and in_channels == out_channels else
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(inputs) + self.skip(inputs), inplace=True)


class GridDetector(nn.Module):
    """Shared grid head: objectness, xywh and ten class logits per slot."""

    def __init__(self, backbone: nn.Module, channels: int):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(GRID_SIZE)
        self.head = nn.Conv2d(
            channels, BOXES_PER_CELL * (5 + len(DET_CLASSES)), kernel_size=1
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.head(self.pool(self.backbone(images)))
        batch, _, rows, columns = output.shape
        return output.view(
            batch, BOXES_PER_CELL, 5 + len(DET_CLASSES), rows, columns
        )


def build_cnn_detector(model_id: str) -> GridDetector:
    """Build either the four-block simple CNN or deeper residual CNN."""
    if model_id == "simple-cnn":
        backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        return GridDetector(backbone, 256)
    if model_id == "complex-cnn":
        backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )
        return GridDetector(backbone, 512)
    raise ValueError(f"unknown custom CNN model: {model_id}")


class GridDetectionDataset(Dataset):
    """Load COCO boxes and encode at most five objects per grid cell."""

    def __init__(self, images_dir: Path, annotation: Path, train: bool = False):
        data = json.loads(annotation.read_text(encoding="utf-8"))
        self.images_dir = images_dir
        self.images = data["images"]
        self.train = train
        self.by_image = defaultdict(list)
        for item in data["annotations"]:
            self.by_image[item["image_id"]].append(item)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        meta = self.images[index]
        width, height = meta["width"], meta["height"]
        with Image.open(self.images_dir / meta["file_name"]) as source:
            image = source.convert("RGB")

        annotations = self.by_image[meta["id"]]
        flip = self.train and random.random() < 0.5
        if flip:
            image = TF.hflip(image)
        image = TF.to_tensor(image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR))

        rows, columns = GRID_SIZE
        objectness = torch.zeros(BOXES_PER_CELL, rows, columns)
        boxes = torch.zeros(BOXES_PER_CELL, 4, rows, columns)
        classes = torch.full((BOXES_PER_CELL, rows, columns), -1, dtype=torch.long)

        for annotation in annotations:
            x, y, box_width, box_height = annotation["bbox"]
            if flip:
                x = width - x - box_width
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            column = min(columns - 1, max(0, int(center_x * columns)))
            row = min(rows - 1, max(0, int(center_y * rows)))
            free = (objectness[:, row, column] == 0).nonzero(as_tuple=False)
            if not len(free):
                continue
            slot = int(free[0])
            objectness[slot, row, column] = 1
            boxes[slot, :, row, column] = torch.tensor([
                center_x * columns - column,
                center_y * rows - row,
                box_width / width,
                box_height / height,
            ])
            classes[slot, row, column] = int(annotation["category_id"]) - 1

        target = {
            "objectness": objectness,
            "boxes": boxes,
            "classes": classes,
            "image_id": torch.tensor(meta["id"]),
            "width": torch.tensor(width),
            "height": torch.tensor(height),
        }
        return image, target


def detection_loss(output: torch.Tensor, target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Balanced objectness plus box and class losses."""
    object_logits = output[:, :, 0]
    positive = target["objectness"].bool()
    negative = ~positive
    zero = output.sum() * 0
    positive_object = (
        F.binary_cross_entropy_with_logits(object_logits[positive], torch.ones_like(object_logits[positive]))
        if positive.any() else zero
    )
    negative_object = (
        F.binary_cross_entropy_with_logits(object_logits[negative], torch.zeros_like(object_logits[negative]))
        if negative.any() else zero
    )
    predicted_boxes = output[:, :, 1:5].sigmoid().permute(0, 1, 3, 4, 2)
    expected_boxes = target["boxes"].permute(0, 1, 3, 4, 2)
    box_loss = F.smooth_l1_loss(predicted_boxes[positive], expected_boxes[positive]) if positive.any() else zero
    class_logits = output[:, :, 5:].permute(0, 1, 3, 4, 2)
    class_loss = F.cross_entropy(class_logits[positive], target["classes"][positive]) if positive.any() else zero
    losses = {
        "object": positive_object + 0.1 * negative_object,
        "box": 5.0 * box_loss,
        "class": class_loss,
    }
    losses["total"] = sum(losses.values())
    return losses


def decode_detections(
    output: torch.Tensor,
    image_id: int,
    width: int,
    height: int,
    confidence: float = 0.05,
    iou: float = 0.7,
    max_detections: int = 100,
) -> list[dict]:
    """Decode one grid output into COCO detections and apply class-aware NMS."""
    slots, _, rows, columns = output.shape
    objectness = output[:, 0].sigmoid()
    class_probability, class_index = output[:, 5:].softmax(1).max(1)
    scores = objectness * class_probability
    keep_mask = scores >= confidence
    if not keep_mask.any():
        return []

    slot_index, row_index, column_index = keep_mask.nonzero(as_tuple=True)
    raw_boxes = output[slot_index, 1:5, row_index, column_index].sigmoid()
    center_x = (column_index + raw_boxes[:, 0]) / columns * width
    center_y = (row_index + raw_boxes[:, 1]) / rows * height
    box_width = raw_boxes[:, 2] * width
    box_height = raw_boxes[:, 3] * height
    boxes = torch.stack([
        center_x - box_width / 2,
        center_y - box_height / 2,
        center_x + box_width / 2,
        center_y + box_height / 2,
    ], dim=1)
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    selected_scores = scores[keep_mask]
    selected_classes = class_index[keep_mask]
    keep = batched_nms(boxes, selected_scores, selected_classes, iou)[:max_detections]

    records = []
    for index in keep.tolist():
        x1, y1, x2, y2 = boxes[index].tolist()
        records.append({
            "image_id": int(image_id),
            "category_id": int(selected_classes[index]) + 1,
            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            "score": round(float(selected_scores[index]), 5),
        })
    return records


@torch.no_grad()
def export_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    output_path: Path,
    confidence: float = 0.05,
    iou: float = 0.7,
    max_detections: int = 100,
) -> Path:
    """Run a custom CNN and write one common COCO prediction file."""
    model.eval()
    records = []
    for images, targets in loader:
        outputs = model(images.to(device, non_blocking=True)).cpu()
        for index, output in enumerate(outputs):
            records.extend(decode_detections(
                output,
                int(targets["image_id"][index]),
                int(targets["width"][index]),
                int(targets["height"][index]),
                confidence,
                iou,
                max_detections,
            ))
    return write_coco_predictions(records, output_path)
