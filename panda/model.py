import importlib
import os
import sys

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .logging import get_logger
from .model_base import PointTransformerV3, flash_attn as _flash_attn
from .model_panoptic import Detector
from .model_segment import Segmenter
from .utils import filter_kwargs, filter_state_dict, set_flash_attention

logger = get_logger(__name__)

MODELS = ["base", "particle", "interaction", "semantic"]

def load(
    name: str = "pretrain",
    download_root: str | None = None,
    repo_id: str = "deeplearnphysics/panda",
    custom_config: dict | None = None,
    custom_cls: nn.Module | None = None,
):
    """Load a model checkpoint from HuggingFace or a local path."""
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

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt.get("config", None) # if none will be loaded from local code

    if custom_cls is not None:
        model_cls = custom_cls
    elif name == "base":
        model_cls = PointTransformerV3
    elif name in {"particle", "interaction"}:
        model_cls = Detector
    elif name == "semantic":
        model_cls = Segmenter
    else: # load directly from exp pimm directory
        cfg_dir = os.path.dirname(os.path.dirname(ckpt_path))
        assert os.path.exists(cfg_dir), "exp path containing codebase must exist if providing raw weights"
        code_path = f"{cfg_dir}/code"
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))
        if str(cfg_dir) not in sys.path:
            sys.path.insert(0, str(cfg_dir))
        from pimm.models.builder import MODELS as PIMM_MODELS
        config = importlib.import_module("config").model
        if "backbone" in config:
            config = config["backbone"]
        model_cls = PIMM_MODELS.get(config['type'])
        ckpt["state_dict"] = filter_state_dict(ckpt["state_dict"])

    if custom_config is not None:
        for key, value in custom_config.items():
            config[key] = value
        # Segmenter and Detector keep PointTransformerV3's configuration in a
        # nested ``backbone`` mapping.  Preserve the flat override supported by
        # base checkpoints while applying it to wrapped pretrained models too.
        if "enable_flash" in custom_config:
            set_flash_attention(config, custom_config["enable_flash"])

    # disable fa if not installed/turned off in config
    if _flash_attn is None and set_flash_attention(config, False):
        logger.warning(
            "FlashAttention is not installed; using native PyTorch attention. "
            "Install panda[flash] to enable FlashAttention if you have an Ampere+ GPU."
        )

    config, _ = filter_kwargs(model_cls, config)
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
