"""Faster R-CNN dataset and model construction.

Resolution matching: Ultralytics imgsz=640 letterboxes a 1280x720 image to
640x360. torchvision's default (min_size=800) would instead feed roughly
1422x800, giving Faster R-CNN ~8x the pixels and making any comparison
meaningless. min_size/max_size below reproduce YOLO's 640x360 exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from .constants import DET_CLASSES

# Matches YOLO imgsz=640 on 1280x720 source imagery.
MIN_SIZE = 360
MAX_SIZE = 640


class CocoDetectionDataset(Dataset):
    """Reads the project's COCO json; returns torchvision detection targets."""

    def __init__(self, images_dir: Path, ann_file: Path, train: bool = False):
        self.images_dir = Path(images_dir)
        self.train = train
        data = json.loads(Path(ann_file).read_text(encoding="utf-8"))
        self.images = data["images"]
        self.by_image: dict[int, list] = {im["id"]: [] for im in self.images}
        for a in data["annotations"]:
            self.by_image[a["image_id"]].append(a)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        meta = self.images[i]
        img = Image.open(self.images_dir / meta["file_name"]).convert("RGB")
        anns = self.by_image[meta["id"]]

        boxes = [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        labels = [a["category_id"] for a in anns]  # already 1-based; 0 is background

        tensor = TF.to_tensor(img)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.train and torch.rand(1).item() < 0.5:  # horizontal flip
            tensor = tensor.flip(-1)
            w = tensor.shape[-1]
            if boxes_t.numel():
                boxes_t = boxes_t[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) \
                    + torch.tensor([w, 0, w, 0])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(meta["id"]),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN R50-FPN v2 with a resized classifier head."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights, min_size=MIN_SIZE, max_size=MAX_SIZE
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(DET_CLASSES) + 1)
    return model


@torch.no_grad()
def predict_to_coco(model, loader, device, out_json: Path) -> Path:
    """Write COCO-format detections for every image in `loader`.

    `image_id` is carried through from the dataset, which took it from the
    ground truth file - never from enumeration order. torchvision labels are
    already 1-based (0 is background), matching the COCO category_id
    convention, so no offset is applied here.
    """
    model.eval()
    preds = []
    for images, targets in loader:
        images = [i.to(device, non_blocking=True) for i in images]
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
        for t, o in zip(targets, outputs):
            img_id = int(t["image_id"])
            boxes = o["boxes"].float().cpu().tolist()
            scores = o["scores"].float().cpu().tolist()
            labels = o["labels"].cpu().tolist()
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
                preds.append({
                    "image_id": img_id,
                    "category_id": int(c),
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(float(s), 5),
                })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    return out_json
