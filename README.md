
<div align="center">
  <h2>Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics</h2>
</div>

<div align="center">
<a href="https://arxiv.org/pdf/2512.01324">[Paper]</a>
<a href="https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M">[Dataset]</a>
<a href="https://youngsm.com/panda">[Project Site]</a>
<a href="./notebooks">[Tutorial]</a>
<a href="#citing-panda">[BibTeX]</a>
</div>

This repo provides pre-trained models, inference code, and visualization demos for LArTPC point cloud analysis with Panda, a sensor-level foundation model for LArTPC point cloud analysis. The training and evaluation code can be found in the [`pimm`](https://github.com/DeepLearnPhysics/pimm) repository.

<div align='left'>
<img src="https://youngsm.com/assets/img/panda/teaser_full.png" alt="teaser" width="800" />
</div>

## Overview
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Citation](#citation)

## Installation
This repo provides two ways of installation: **standalone mode** and **package mode**.
The provided GPU environments assume Python 3.10, Linux x86-64, and CUDA 12.x.

- The **standalone mode** is recommended for quick inference, visualization,
  and running the demo notebooks.

  <details open>
  <summary><b>uv</b></summary>

  If you haven't already, install uv via running `curl -LsSf https://astral.sh/uv/install.sh | sh`. Then,

  ```bash
  git clone https://github.com/DeepLearnPhysics/Panda.git && cd Panda  
  uv sync

  # optionally with Flash Attention (Ampere+ GPUs)
  uv sync --extra flash
  ```

  </details>

  <details>
  <summary><b>Conda</b></summary>

  ```bash
  # run `unset CUDA_PATH` first if it points to another CUDA installation.
  conda env create -f environment.yml --verbose
  conda activate panda
  ```

  </details>

- The **package mode** installs Panda directly into the current environment
  from GitHub.

  <details open>
  <summary><b>uv</b></summary>

  ```bash
  uv pip install "panda @ git+https://github.com/DeepLearnPhysics/Panda.git"

  # optionally with FlashAttention (Ampere+ GPUs)
  uv pip install "panda[flash] @ git+https://github.com/DeepLearnPhysics/Panda.git"
  ```

  </details>

  <details>
  <summary><b>Conda</b></summary>

  ```bash
  conda activate <environment>
  pip install "panda @ git+https://github.com/DeepLearnPhysics/Panda.git"

  # optionally with FlashAttention (Ampere+ GPUs)
  pip install "panda[flash] @ git+https://github.com/DeepLearnPhysics/Panda.git"
  ```

  </details>

## Dataset
We use the **PILArNet-M** dataset (~168 GB), which can be downloaded directly from HuggingFace:

```python
import panda

# auto-download and create dataset
# dataset is cached in `data_root`, which defaults to `~/.cache/pilarnet`
dataset = panda.PILArNetH5Dataset(split="all", data_root=...)

# or download manually first
data_root = panda.download_pilarnet(split="all", data_root=...)
```

See **[DATASET.md](DATASET.md)** for full documentation on dataset structure, labels, and more advanced usage.

## Quick Start
- **Model.** Load the pre-trained model by running the following command:
  ```python
  # load the pre-trained model from Huggingface
  # supported models: "base", "particle", "interaction", "semantic"
  # ckpt is cached in ~/.cache/panda/ckpt, and the path can be customized by setting 'download_root'
  import panda

  model = panda.load("base").cuda()
  
  # load the pre-trained model from local path
  # assume the ckpt file is stored in the 'ckpt' folder
  model = panda.load("ckpt/panda_base.pth").cuda()
  
  # the ckpt file stores the config and state_dict of pretrained model
  ```
  If *FlashAttention* is not available, you will get a warning message and the model will use native PyTorch attention. You can also manually set the `enable_flash` flag to False in the config file to disable FlashAttention:

  ```python
  model = panda.load("base", custom_config=dict(enable_flash=False)).cuda()
  ```
- **Inference.** Run the inference by running the following command:
  ```python
  >>> EVENT_IDX = 0
  >>> dataset = panda.PILArNetH5Dataset(split="test") # downloads ~7GB
  >>> point = dataset[EVENT_IDX]
  >>> for key in point.keys():
      if isinstance(point[key], torch.Tensor):
          point[key] = point[key].cuda(non_blocking=True)
  >>> point = model(point)
  >>> point.coord.shape, point.feat.shape, point.offset.shape
  (torch.Size([1175, 3]), torch.Size([1175, 1232]), torch.Size([1]))
  ```

  Full example notebooks for accessing the dataset, image encoding, particle and interaction clustering, and semantic segmentation can be found in [notebooks](./notebooks).

## Citing Panda

If you find this work useful, please consider citing the following paper:

```bibtex
@misc{young2025pandaselfdistillationreusablesensorlevel,
      title={Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics}, 
      author={Samuel Young and Kazuhiro Terao},
      year={2025},
      eprint={2512.01324},
      archivePrefix={arXiv},
      primaryClass={hep-ex},
      url={https://arxiv.org/abs/2512.01324}, 
}
```


## Acknowledgements

This repository is based on the Sonata paper's inference repository, which can be found [https://github.com/facebookresearch/sonata](here). Parts of this code, of which were taken from the original repository, are licensed under the [Apache 2.0 license](LICENSE).

This work is supported by the U.S. Department of Energy, Office of Science, and Office of High Energy Physics under Contract No. DE-AC02-76SF00515.

## Contact

Sam Young [youngsam@stanford.edu](mailto:youngsam@stanford.edu).
