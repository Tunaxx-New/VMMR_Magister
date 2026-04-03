"""
car_classifier.py - Wrapper around the VMMR checkpoint for car model classification.

Runs on cropped car images produced by YOLOv5 and returns the predicted make/model.

Usage:
    classifier = CarClassifier("clip_cars_checkpoint.pth")
    label, confidence = classifier.predict_crop(crop_bgr)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ── Model factory (copied from user's predict.py) ────────────────────────────

def _pick_weights(default_enum_name):
    try:
        return getattr(models, default_enum_name).IMAGENET1K_V1
    except Exception:
        return None


def _initialize_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()

    if name == "alexnet":
        model = models.alexnet(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "vgg19":
        model = models.vgg19(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "resnet":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name in ["efficientnet", "efficientnet_b0"]:
        model = models.efficientnet_b0(weights=_pick_weights("EfficientNet_B0_Weights"))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name in ["mobilenet", "mobilenet_v3", "mobilenet_v3_large"]:
        model = models.mobilenet_v3_large(weights=_pick_weights("MobileNet_V3_Large_Weights"))
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif name in ["vit", "vit_b_16", "vit_base"]:
        model = models.vit_b_16(weights=_pick_weights("ViT_B_16_Weights"))
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif name == "convnext":
        model = models.convnext_base(weights=_pick_weights("ConvNeXt_Base_Weights"))
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif name in ["simclrv2", "simclr", "simclr_v2"]:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "clip":
        import clip

        class CLIPClassifier(nn.Module):
            def __init__(self, clip_model, embed_dim, num_classes):
                super().__init__()
                self.visual   = clip_model.visual
                self.head     = nn.Linear(embed_dim, num_classes)

            def forward(self, x):
                # visual.forward returns the projected embedding (float16 on GPU)
                feats = self.visual(x.half() if next(self.visual.parameters()).dtype == torch.float16 else x)
                return self.head(feats.float())

        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        embed_dim = clip_model.visual.proj.shape[1]   # 512 for ViT-B/32
        model = CLIPClassifier(clip_model, embed_dim, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name!r}")

    return model


def _load_simclr_weights(model: nn.Module, state_dict: dict) -> None:
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_state[k.replace("encoder.", "")] = v
        elif k.startswith("backbone."):
            new_state[k.replace("backbone.", "")] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    logger.info("SimCLR weights | missing=%d unexpected=%d", len(missing), len(unexpected))


# ── Transform ─────────────────────────────────────────────────────────────────

def _get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Main class ────────────────────────────────────────────────────────────────

class CarClassifier:
    """
    Loads a VMMR checkpoint and classifies car crops.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint saved by your training script.
        The checkpoint must contain keys:
            model_name, num_classes, class_names, model_state_dict
    device : str | None
        'cuda', 'cpu', or None for auto-detect.
    min_crop_size : int
        Minimum width OR height of a crop to run inference on.
        Crops smaller than this are skipped (too blurry to classify).
    confidence_threshold : float
        Minimum softmax confidence to log a result. Below this → logged as "uncertain".
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        min_crop_size: int = 48,
        confidence_threshold: float = 0.30,
    ):
        self.min_crop_size         = min_crop_size
        self.confidence_threshold  = confidence_threshold
        self.transform             = _get_transform()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("Loading VMMR checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model_name  = ckpt["model_name"]
        self.num_classes = ckpt["num_classes"]
        self.class_names = ckpt["class_names"]

        model = _initialize_model(self.model_name, self.num_classes)

        state_dict = ckpt["model_state_dict"]

        if self.model_name.lower() in ("simclrv2", "simclr", "simclr_v2"):
            _load_simclr_weights(model, state_dict)
        elif self.model_name.lower() == "clip":
            # Checkpoint may have been saved from the raw CLIPClassifier wrapper
            # or from the bare clip_model — try strict first, fall back to loose
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("CLIP load: missing=%d unexpected=%d — trying key remap",
                               len(missing), len(unexpected))
                # Remap: keys without prefix -> visual.* prefix
                remapped = {}
                for k, v in state_dict.items():
                    if k.startswith("visual."):
                        remapped[k] = v
                    elif k.startswith("head."):
                        remapped[k] = v
                    else:
                        remapped[f"visual.{k}"] = v
                missing2, unexpected2 = model.load_state_dict(remapped, strict=False)
                logger.info("CLIP remap: missing=%d unexpected=%d", len(missing2), len(unexpected2))
        else:
            model.load_state_dict(state_dict)

        self.model = model.to(self.device).eval()
        logger.info(
            "Classifier ready | model=%s  classes=%d  device=%s",
            self.model_name, self.num_classes, self.device,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict_crop(
        self,
        crop_bgr: np.ndarray,
    ) -> tuple[str, float] | None:
        """
        Classify a single BGR crop (from OpenCV / YOLO detection).

        Returns
        -------
        (class_name, confidence) if the crop is large enough and confidence
        meets the threshold, otherwise None.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if min(h, w) < self.min_crop_size:
            logger.debug("Crop too small (%dx%d) — skipping classifier", w, h)
            return None

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        confidence  = conf.item()
        class_label = self.class_names[pred.item()]

        if confidence < self.confidence_threshold:
            logger.debug("Low confidence %.2f for %s — skipping", confidence, class_label)
            return None

        return class_label, confidence

    def predict_with_tta(
        self,
        crop_bgr: np.ndarray,
        augments: int = 4,
    ) -> tuple[str, float] | None:
        """
        Test-Time Augmentation: run the classifier on multiple crops
        (original + flipped + brightness variants) and average the softmax.

        More accurate than predict_crop() at ~3-4x the compute cost.
        Only call this when you have time budget (e.g. on stable tracks).

        augments: number of augmented views (1=original only, up to 6)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if min(h, w) < self.min_crop_size:
            return None

        import torchvision.transforms.functional as TF

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        views = [pil]
        if augments >= 2:
            views.append(TF.hflip(pil))
        if augments >= 3:
            views.append(TF.adjust_brightness(pil, 1.3))
        if augments >= 4:
            views.append(TF.adjust_contrast(pil, 1.2))
        if augments >= 5:
            views.append(TF.adjust_brightness(pil, 0.75))
        if augments >= 6:
            views.append(TF.rotate(pil, 5))

        tensors = torch.stack(
            [self.transform(v) for v in views]
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensors)
            # Average softmax across all augmented views
            avg_probs = torch.softmax(outputs, dim=1).mean(dim=0)
            conf, pred = avg_probs.max(0)

        confidence  = conf.item()
        class_label = self.class_names[pred.item()]

        if confidence < self.confidence_threshold:
            return None
        return class_label, confidence

    def predict_top_k(
        self,
        crop_bgr: np.ndarray,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Return top-k predictions as [(class_name, confidence), ...].
        Returns empty list if crop is too small or all confidences are low.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        h, w = crop_bgr.shape[:2]
        if min(h, w) < self.min_crop_size:
            return []

        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs   = self.model(tensor)
            probs     = torch.softmax(outputs, dim=1)[0]
            top_probs, top_idxs = probs.topk(min(k, self.num_classes))

        return [
            (self.class_names[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_idxs)
            if prob.item() >= self.confidence_threshold
        ]