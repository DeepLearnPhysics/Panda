# PILArNet-M Dataset

Panda uses the **PILArNet-M** dataset for training and evaluation. The dataset can be downloaded directly from HuggingFace and is organized by splits (`train`, `val`, `test`).

> NOTE: This is a large dataset (~168 GB total). Make sure you have sufficient disk space before downloading.

## Dataset Size

| Split | Events | Size |
|-------|--------|------|
| Train | 1,082,400 | 151 GB |
| Validation | 66,800 | 9.34 GB |
| Test | 50,000 | 7.02 GB |
| **Total** | **1,199,200** | **~167 GB** |

## Dataset Structure

Each split contains:
- **H5 files** (`*.h5`): Point cloud data with coordinates, energy deposits, and labels
- **Index files** (`*_points.npy`): Pre-computed point counts for fast dataset initialization

### Labels

The dataset contains the following semantic (motif) classes:
| ID | Class |
|----|-------|
| 0 | Shower |
| 1 | Track |
| 2 | Michel |
| 3 | Delta |
| 4 | Low energy deposit |

And the following particle ID (PID) classes:
| ID | Particle |
|----|----------|
| 0 | Photon |
| 1 | Electron |
| 2 | Muon |
| 3 | Pion |
| 4 | Proton |
| 5 | None (Low energy deposit) |

## Quick Download

```python
import panda

# download train split (default)
data_root = panda.download_pilarnet(split="train")

# download specific splits
data_root = panda.download_pilarnet(split=["train", "val", "test"])

# download all data
data_root = panda.download_pilarnet(split=None)
```

## Using with PILArNetH5Dataset

```python
import panda

# option 1: auto-download from HuggingFace (recommended)
dataset = panda.PILArNetH5Dataset(split="train")

# option 2: specify custom HuggingFace repo
dataset = panda.PILArNetH5Dataset(
    split="train",
    hf_repo_id="deeplearnphysics/pilarnet-m",
    hf_local_dir="~/.cache/pilarnet",
)

# option 3: use local files (skip download)
dataset = panda.PILArNetH5Dataset(
    data_root="/path/to/local/data",
    split="train",
)
```

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_root` | str | None | Local path to data. If None, downloads from HF. |
| `split` | str | "train" | Dataset split: "train", "val", or "test" |
| `energy_threshold` | float | 0.0 | Minimum energy threshold (MeV) |
| `min_points` | int | 1024 | Minimum points per event |
| `hf_repo_id` | str | "deeplearnphysics/pilarnet-m" | HuggingFace repository ID |
| `hf_local_dir` | str | "~/.cache/pilarnet" | Local cache directory |

### Accessing Data

```python
import panda
import numpy as np

dataset = panda.PILArNetH5Dataset(split="val", energy_threshold=0.13)

# get a single event
data = dataset[0]

# available keys
print(data.keys())
# dict_keys(['coord', 'grid_coord', 'energy', 'segment_motif', 
#            'segment_particle', 'instance_particle', 'instance_interaction', ...])

# coordinates: (N, 3)
coords = data['coord']

# energy deposits: (N, 1)  
energy = data['energy']

# semantic labels: (N, 1)
motif_labels = data['segment_motif']

# particle ID labels: (N, 1)
pid_labels = data['segment_particle']

# particle instance IDs: (N, 1)
instance_labels = data['instance_particle']
```

## Advanced: PILArNetHFInterface

For more control over downloads, use the interface directly:

```python
from panda import PILArNetHFInterface

interface = PILArNetHFInterface(
    repo_id="deeplearnphysics/pilarnet-m",
    revision="main",
    local_dir="~/.cache/pilarnet",
)

# list available files
files = interface.list_files("**/*.h5")

# download specific split
interface.download_split("train")

# download all data
interface.download_all()
```

## Using with DataLoader

```python
from torch.utils.data import DataLoader
import panda

dataset = panda.PILArNetH5Dataset(split="train", energy_threshold=0.13)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=panda.utils.collate_fn,
    num_workers=4,
)

for batch in dataloader:
    # batch is a dict with concatenated tensors
    coords = batch['coord']  # (total_points, 3)
    offset = batch['offset']  # batch boundaries
    ...
```

