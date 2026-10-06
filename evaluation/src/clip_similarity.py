from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


_CLIP_MODEL_CACHE = {}


def get_clip_model_and_processor(model_name: str):
    cached = _CLIP_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    _CLIP_MODEL_CACHE[model_name] = (model, processor, device)
    return model, processor, device


def extract_feature_tensor(image_features):
    pooler_output = getattr(image_features, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    if isinstance(image_features, torch.Tensor):
        return image_features

    if isinstance(image_features, (tuple, list)):
        tensor_candidates = [item for item in image_features if isinstance(item, torch.Tensor)]
        for item in tensor_candidates:
            if item.ndim == 2:
                return item
        if tensor_candidates:
            return tensor_candidates[0]

    raise TypeError(f"Unexpected image feature type: {type(image_features)!r}")


def _load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def clip_similarity_batch(image_pairs: Sequence[Tuple[str | Path, str | Path]]) -> List[float]:
    if not image_pairs:
        return []

    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model, clip_processor, device = get_clip_model_and_processor(clip_model_name)

    images: List[Image.Image] = []
    for gt_path, gen_path in image_pairs:
        images.append(_load_rgb_image(gt_path))
        images.append(_load_rgb_image(gen_path))

    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    image_features = extract_feature_tensor(image_features)
    image_features = F.normalize(image_features, dim=-1)

    scores: List[float] = []
    for pair_index in range(len(image_pairs)):
        start = pair_index * 2
        feat_gt = image_features[start : start + 1]
        feat_gen = image_features[start + 1 : start + 2]
        scores.append(F.cosine_similarity(feat_gt, feat_gen).item())

    return scores


def clip_similarity(gt_path, gen_path):
    return clip_similarity_batch([(gt_path, gen_path)])[0]


def clear_clip_model_cache() -> None:
    for model, _, _ in _CLIP_MODEL_CACHE.values():
        if hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception:
                pass

    _CLIP_MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
