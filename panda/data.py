import os
import numpy as np
from .logging import get_logger
from .hf import PILArNetHFInterface
logger = get_logger(__name__)


"""
PILArNet Dataset

This module handles the PILArNet dataset for particle physics point cloud segmentation.
"""

import glob
import h5py
from copy import deepcopy
from torch.utils.data import Dataset

from .transform import Compose, TRANSFORMS

class PILArNetH5Dataset(Dataset):
    """
    PILArNet Dataset that loads directly from h5 files, avoiding the need for preprocessing to individual files.

    The dataset contains the following semantic classes:
    - 0: Shower
    - 1: Track
    - 2: Michel
    - 3: Delta
    - 4: Low energy deposit

    and the following PID classes:
    - 0: Photon
    - 1: Electron
    - 2: Muon
    - 3: Pion
    - 4: Proton
    - 5: None (Low energy deposit)

    HuggingFace Integration:
        Set `data_root=None` and provide `hf_repo_id` to download from HuggingFace.
        Alternatively, pass a `hf_interface` instance for more control.
    """

    def __init__(
        self,
        data_root=None,
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        energy_threshold=0.0,
        min_points=1024,
        max_len=-1,
        remove_low_energy_scatters=False,
        copy=None,
        # HuggingFace parameters
        hf_repo_id: str | None = None,
        hf_revision: str | None = "main",
        hf_token: str | bool | None = None,
        hf_cache_dir: str | None = None,
        hf_local_dir: str | None = None,
        hf_interface: PILArNetHFInterface | None = None,
        auto_download: bool = True,
    ):
        super().__init__()
        self.split = split
        self.transform = Compose(transform if transform else default_transform(copy=copy))
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.loop = loop if not test_mode else 1
        self.ignore_index = ignore_index

        # Initialize HF interface and resolve data_root
        self.hf_interface = hf_interface
        if data_root is None or hf_repo_id is not None:
            if self.hf_interface is None:
                self.hf_interface = PILArNetHFInterface(
                    repo_id=hf_repo_id,
                    revision=hf_revision,
                    token=hf_token,
                    cache_dir=hf_cache_dir,
                    local_dir=hf_local_dir,
                )
            if auto_download:
                logger.info(f"Downloading PILArNet data from HuggingFace: {self.hf_interface.repo_id}")
                self.data_root = self.hf_interface.get_data_root(split=split)
            else:
                self.data_root = str(self.hf_interface.local_dir)
        else:
            self.data_root = data_root

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # PILArNet specific parameters
        self.energy_threshold = energy_threshold
        self.min_points = min_points
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.max_len = max_len
        # Get list of h5 files
        self.h5_files = self.get_h5_files()
        assert len(self.h5_files) > 0, f"No h5 files found in {self.data_root}"
        self.initted = False
        self.file_events = []

        # Build index for faster access
        self._build_index()

        logger.info(
            "Total number of samples in PILArNet {} set: {} x {}.".format(
                self.cumulative_lengths[-1], self.loop, split
            )
        )

    def get_h5_files(self):
        """Get list of h5 files based on the split."""
        if isinstance(self.split, str):
            split_pattern = f"*{self.split}/*.h5"
        else:
            split_pattern = [f"*{s}/*.h5" for s in self.split]

        if isinstance(split_pattern, list):
            h5_files = []
            for pattern in split_pattern:
                h5_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        else:
            h5_files = glob.glob(os.path.join(self.data_root, split_pattern))

        return sorted(h5_files)

    def _build_index(self):
        """Build an index of valid point clouds for faster access."""
        logger.info("Building index for PILArNetH5Dataset")

        self.cumulative_lengths = []
        self.indices = []

        for h5_file in self.h5_files:
            try:
                # Check if points count file exists
                points_file = h5_file.replace(".h5", "_points.npy")
                if os.path.exists(points_file):
                    npoints = np.load(points_file)
                    index = np.argwhere(npoints >= self.min_points).flatten()
                else:
                    # No points file, count on the fly
                    logger.info(
                        f"No points count file for {h5_file}, counting points on the fly"
                    )
                    with h5py.File(h5_file, "r", libver="latest", swmr=True) as f:
                        # Get all point counts
                        npoints = []
                        for i in range(f["cluster"].shape[0]):
                            cluster_size = f["cluster"][i].reshape(-1, 6)[:, 0]
                            npoints.append(cluster_size.sum())
                        npoints = np.array(npoints)
                        index = np.argwhere(npoints >= self.min_points).flatten()
                        self.file_events.append(npoints.shape[0])
                if os.path.exists(points_file):
                    self.file_events.append(int(npoints.shape[0]))
            except Exception as e:
                logger.warning(f"Error processing {h5_file}: {e}")
                index = np.array([])
                self.file_events.append(0)

            self.cumulative_lengths.append(index.shape[0])
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        logger.info(
            f"Found {self.cumulative_lengths[-1]} point clouds with at least {self.min_points} points"
        )

    def h5py_worker_init(self):
        """Initialize h5py files for each worker."""
        self.h5data = []
        for h5_file in self.h5_files:
            self.h5data.append(h5py.File(h5_file, mode="r", libver="latest", swmr=True))
        self.initted = True

    def get_data(self, idx):
        """Load a point cloud from h5 file.

        Output dictionary:
        - coord: (N, 3) array of coordinates
        - energy: (N, 1) array of energies
        - momentum: (N, 1) array of particle momentum
        - vertex: (N, 3) array of vertices
        - segment_motif: (N, 1) array of motif labels
        - segment_pid: (N, 1) array of PID labels
        - instance_particle: (N, 1) array of particle instance labels
        - instance_interaction: (N, 1) array of interaction instance labels
        - segment_interaction: (N, 1) array of interaction labels
        """
        if not self.initted:
            self.h5py_worker_init()

        # Find which h5 file and index the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if h5_idx > 0:
            idx_in_file = idx - self.cumulative_lengths[h5_idx - 1]
        else:
            idx_in_file = idx

        h5_file = self.h5data[h5_idx]
        file_idx = self.indices[h5_idx][idx_in_file]

        # Load point cloud data
        data = h5_file["point"][file_idx].reshape(-1, 8)[:, [0, 1, 2, 3]]  # (x,y,z,e)
        cluster_size, group_id, interaction_id, semantic_id, pid = (
            h5_file["cluster"][file_idx].reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
        )
        mom, vtx_x, vtx_y, vtx_z = (
            h5_file["cluster_extra"][file_idx].reshape(-1, 5)[:, [1, 2, 3, 4]].T
        )

        pid[pid == -1] = 5  # -1 --> 5 (LED/None class)

        # Remove low energy scatters if configured
        if self.remove_low_energy_scatters:
            data = data[cluster_size[0] :]
            semantic_id, group_id, interaction_id, pid, cluster_size = (
                semantic_id[1:],
                group_id[1:],
                interaction_id[1:],
                pid[1:],
                cluster_size[1:],
            )
            mom, vtx_x, vtx_y, vtx_z = mom[1:], vtx_x[1:], vtx_y[1:], vtx_z[1:]

        # Compute semantic ids for each point
        data_semantic_id = np.repeat(semantic_id, cluster_size)
        data_group_id = np.repeat(group_id, cluster_size)
        data_interaction_id = np.repeat(interaction_id, cluster_size)
        data_pid = np.repeat(pid, cluster_size)
        data_mom = np.repeat(mom, cluster_size)
        data_vtx_x = np.repeat(vtx_x, cluster_size)
        data_vtx_y = np.repeat(vtx_y, cluster_size)
        data_vtx_z = np.repeat(vtx_z, cluster_size)

        # Apply energy threshold if needed
        if self.energy_threshold > 0:
            threshold_mask = data[:, 3] > self.energy_threshold
            data = data[threshold_mask]
            data_semantic_id = data_semantic_id[threshold_mask]
            data_group_id = data_group_id[threshold_mask]
            data_interaction_id = data_interaction_id[threshold_mask]
            data_pid = data_pid[threshold_mask]
            data_mom = data_mom[threshold_mask]
            data_vtx_x = data_vtx_x[threshold_mask]
            data_vtx_y = data_vtx_y[threshold_mask]
            data_vtx_z = data_vtx_z[threshold_mask]
            if data.shape[0] < self.min_points:
                # Try another data point if this one is too small after filtering
                return self.get_data((idx + 1) % len(self))

        # Prepare return dictionary
        data_dict = {}

        # Get coordinates
        data_dict["coord"] = data[:, :3].astype(np.float32)

        # Process energy (raw)
        energy = data[:, 3].astype(np.float32)
        data_dict["energy"] = energy[:, None]

        # Momentum
        data_dict["momentum"] = data_mom.astype(np.float32)[:, None]
        data_dict["vertex"] = np.stack(
            [data_vtx_x, data_vtx_y, data_vtx_z], axis=1
        ).astype(np.float32)

        # Get semantic labels
        data_dict["segment_motif"] = data_semantic_id.astype(np.int32)[:, None]
        data_dict["segment_particle"] = data_pid.astype(np.int32)[:, None]
        # compute both particle- and interaction-level instances
        particle_ids = data_group_id.astype(np.int32)
        interaction_ids = data_interaction_id.astype(np.int32)

        def map_instance_ids(instance_ids_array):
            """Map instance ids to new ids.

            i.e. instead of having instance ids like [0, 1, 23, 47, 52, 53, 54, 55, 56, 57],
                 we want to have instance ids like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            """
            unique_ids_local = np.unique(instance_ids_array)
            id_mapping_local = {
                old_id: new_id
                for new_id, old_id in enumerate(unique_ids_local[unique_ids_local >= 0])
            }
            return np.array(
                [id_mapping_local.get(id_val, -1) for id_val in instance_ids_array],
                dtype=np.int32,
            )[:, None]

        instance_particle = map_instance_ids(particle_ids)
        instance_interaction = map_instance_ids(interaction_ids)

        # always return both flavors
        data_dict["instance_particle"] = instance_particle
        data_dict["instance_interaction"] = instance_interaction
        data_dict["segment_interaction"] = (interaction_ids[:, None] != -1).astype(
            np.int32
        )  # 1 if not background, 0 if background

        # Add metadata
        h5_name = os.path.basename(self.h5_files[h5_idx])
        data_dict["name"] = f"{h5_name}_{file_idx}"
        data_dict["split"] = self.split if isinstance(self.split, str) else "custom"

        return data_dict

    def get_data_name(self, idx):
        """Get name for the point cloud."""
        if not self.initted:
            self.h5py_worker_init()

        # Find which h5 file and index the point cloud is in
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if h5_idx > 0:
            idx_in_file = idx - self.cumulative_lengths[h5_idx - 1]
        else:
            idx_in_file = idx

        h5_name = os.path.basename(self.h5_files[h5_idx])
        file_idx = self.indices[h5_idx][idx_in_file]

        return f"{h5_name}_{file_idx}"

    def prepare_train_data(self, idx):
        """Prepare training data with transforms."""
        data_dict = self.get_data(idx % len(self))
        return self.transform(data_dict)

    def prepare_test_data(self, idx):
        """Prepare test data with test transforms."""
        # Load data
        data_dict = self.get_data(idx % len(self))

        # Apply transforms
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # Test mode specific handling
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = self.get_queries(fragment_list)
        return result_dict

    def __getitem__(self, idx):
        real_idx = idx % len(self)
        if self.test_mode:
            return self.prepare_test_data(real_idx)
        else:
            return self.prepare_train_data(real_idx)

    def __len__(self):
        if self.max_len > 0:
            return min(self.max_len, self.cumulative_lengths[-1]) * self.loop
        return self.cumulative_lengths[-1] * self.loop

    def __del__(self):
        """Clean up open h5 files."""
        if hasattr(self, "initted") and self.initted:
            for h5_file in self.h5data:
                h5_file.close()


def default_transform(copy=None):
    grid_size = 0.001
    transforms = [
        dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
        dict(type="LogTransform", min_val=1.0e-2, max_val=20.0),
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="Copy", keys_dict=copy) if copy else None,
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "instance_particle",
                  "instance_interaction", "energy", "segment_particle",
                  "segment_motif", "segment_interaction"
                ) + tuple(copy.values() if copy else []),
            feat_keys=("coord", "energy",),
        ),
    ]
    return transforms


