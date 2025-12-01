import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from packaging import version

from .logging import get_logger
from .model_base import PointTransformerV3
from .model_panoptic import Detector
from .model_segment import Segmenter
from .utils import filter_kwargs

logger = get_logger(__name__)

MODELS = ["base", "particle", "interaction", "semantic"]

def load(
    name: str = "pretrain",
    download_root: str = None,
    repo_id: str = "deeplearnphysics/panda",
    custom_config: dict = None,
    custom_cls: nn.Module = None,
):
    if name in MODELS:
        logger.info(f"Loading checkpoint from HuggingFace: {name} ...")
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"panda_{name}.pth",
            repo_type="model",
            revision="main",
            local_dir=download_root or os.path.expanduser("~/.cache/panda/ckpt"),
        )
    elif os.path.isfile(name):
        logger.info(f"Loading checkpoint in local path: {name} ...")
        ckpt_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {MODELS}")

    if version.parse(torch.__version__) >= version.parse("2.4"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if custom_config is not None:
        for key, value in custom_config.items():
            ckpt["config"][key] = value

    if custom_cls is not None:
        model_cls = custom_cls
    elif name == "base":
        model_cls = PointTransformerV3
    elif name in {"particle", "interaction"}:
        model_cls = Detector
    elif name == "semantic":
        model_cls = Segmenter

    config, _ = filter_kwargs(model_cls, ckpt["config"])
    model = model_cls(**config)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if len(missing) > 0:
        logger.error(f"Missing keys: {missing}")
        raise ValueError(f"Missing keys: {missing}")
    if len(unexpected) > 0:
        logger.info(f"Unexpected keys: {unexpected}")
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_parameters / 1e6:.2f}M")
    return model
