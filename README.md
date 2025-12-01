
<div align="center">
  <h2>Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics</h2>
</div>

<div align="center">
<a href="#">[Paper]</a>
<a href="https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M">[Dataset]</a>
<a href="https://youngsm.com/panda">[Project Site]</a>
<a href="./notebooks">[Tutorial]</a>
<a href="#citing-panda">[BibTeX]</a>
</div>


This repo provides pre-trained models, inference code, and visualization demos for LArTPC point cloud analysis. The training and evaluation code can be found in the [LAr.FM](https://github.com/DeepLearnPhysics/LAr.FM) repository.

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
- The **standalone mode** is recommended for users who want to use the code for quick inference and visualization. The whole environment including `cuda` and `pytorch` can be easily installed by running the following command:
  ```bash
  # create and activate conda environment named as 'panda'
  # cuda: 12.4, pytorch: 2.5.0
  
  # run `unset CUDA_PATH` if you have installed cuda in your local environment
  conda env create -f environment.yml --verbose
  conda activate panda
  ```

  *We install **FlashAttention** by default, but it is not necessary. If FlashAttention is not available in your local environment, check the Model section in [Quick Start](#quick-start) for a solution.*

- The **package mode** is recommended for users who want to inject the model into a separate codebase. We provide a `setup.py` file for installation. You can install the package by running the following command:
  ```bash
  # ensure CudaCUDA and Pytorch are already installed in your local environment
  
  # CUDA_VERSION: cuda version of local environment (e.g., 124), check by running 'nvcc --version'
  # TORCH_VERSION: torch version of local environment (e.g., 2.5.0), check by running 'python -c "import torch; print(torch.__version__)"'
  pip install spconv-cu${CUDA_VERSION}
  pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu${CUDA_VERSION}.html
  pip install git+https://github.com/Dao-AILab/flash-attention.git
  pip install huggingface_hub timm h5py
  
  # (optional, or directly copy the panda folder to your project)
  python setup.py install
  ```
  Additionally, for running our **demo code**, the following packages are also required:
  ```bash
  pip install plotly matplotlib jupyter
  ```

## Dataset
We use the **PILArNet-M** dataset (~168 GB), which can be downloaded directly from HuggingFace:

```python
import panda

# auto-download and create dataset
dataset = panda.PILArNetH5Dataset(split="all")

# or download manually first
data_root = panda.download_pilarnet(split="all")
```

See **[DATASET.md](DATASET.md)** for full documentation on dataset structure, labels, and more advanced usage.

## Quick Start
- **Model.** Load the pre-trained model by running the following command:
  ```python
  # load the pre-trained model from Huggingface
  # supported models: "base", "particle", "interaction", "semantic"
  # ckpt is cached in ~/.cache/panda/ckpt, and the path can be customized by setting 'download_root'
  model = panda.load("base").cuda()
  
  # load the pre-trained model from local path
  # assume the ckpt file is stored in the 'ckpt' folder
  model = panda.load("ckpt/panda_base.pth").cuda()
  
  # the ckpt file stores the config and state_dict of pretrained model
  ```
  If *FlashAttention* is not available, load the pre-trained model with the following code:
  ```python
  custom_config = dict(enable_flash=False)
  model = panda.load("base", custom_config=custom_config).cuda()
  ```
- **Inference.** Run the inference by running the following command:
  ```python
  EVENT_IDX = 0
  dataset = panda.PILArNetH5Dataset(split="val", energy_threshold=0.13)
  point = dataset[EVENT_IDX]
  for key in point.keys():
      if isinstance(point[key], torch.Tensor):
          point[key] = point[key].cuda(non_blocking=True)
  point = model(point)
  ```

  Full example notebooks for accessing the dataset, image encoding, particle and interaction clustering, and semantic segmentation can be found in [notebooks](./notebooks).

## Citing Panda

If you find this work useful, please consider citing the following paper:

```bibtex
@inproceedings{young2025panda,
    title     = {Panda: Self-distillation of Reusable Sensor-level 
                 Representations for High Energy Physics},
    author    = {Young, Samuel and Terao, Kazuhiro},
    year      = {2025}
}
```


## Acknowledgements

This repository is based on the Sonata paper's inference repository, which can be found [https://github.com/facebookresearch/sonata](here). Parts of this code, of which were taken from the original repository, are licsensed under the [Apache 2.0 license](LICENSE).

This work is supported by the U.S. Department of Energy, Office of Science, and Office of High Energy Physics under Contract No. DE-AC02-76SF00515.

## Contact

Any questions? Any suggestions? Want to collaborate? Feel free to raise an issue on Github or email Sam Young [youngsam@stanford.edu](mailto:youngsam@stanford.edu).
