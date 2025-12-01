import pathlib
from typing import Iterable

from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

from .logging import get_logger

logger = get_logger(__name__)


class PILArNetHFInterface:
    """Interface for downloading PILArNet dataset files from HuggingFace.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., "deeplearnphysics/pilarnet-m").
        revision: Git revision (branch, tag, or commit hash). Defaults to "main".
        token: HF access token. Defaults to locally saved token.
        cache_dir: Path to cache directory.
        local_dir: Path to local directory for downloaded files.
    """

    DEFAULT_REPO_ID = "deeplearnphysics/pilarnet-m"

    def __init__(
        self,
        repo_id: str | None = None,
        revision: str | None = "main",
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
    ) -> None:
        self.repo_id = repo_id or self.DEFAULT_REPO_ID
        self.revision = revision
        self.token = token
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.local_dir = (
            pathlib.Path(local_dir)
            if local_dir
            else pathlib.Path.home() / ".cache" / "pilarnet"
        )
        self._fs = None

    @property
    def fs(self) -> HfFileSystem:
        """Lazy-initialized HuggingFace filesystem."""
        if self._fs is None:
            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def list_files(self, pattern: str = "**/*.h5") -> list[str]:
        """List files in the repository matching a glob pattern."""
        repo_path = f"datasets/{self.repo_id}"
        if self.revision:
            repo_path = f"{repo_path}@{self.revision}"
        full_pattern = f"{repo_path}/{pattern}"
        files = self.fs.glob(full_pattern)
        prefix = f"{repo_path}/"
        return [f.replace(prefix, "") for f in files]

    def download_file(self, filename: str) -> str:
        """Download a single file and return its local path."""
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_dir=str(self.local_dir),
        )

    def download_files(self, filenames: Iterable[str]) -> list[str]:
        """Download multiple files and return their local paths."""
        return [self.download_file(f) for f in filenames]

    def download_split(self, split: str = "train") -> str:
        """Download all h5 files and associated index files for a specific split.

        Returns the local directory path containing the downloaded files.
        """
        h5_pattern = f"*{split}/*.h5"
        h5_files = self.list_files(h5_pattern)

        if not h5_files:
            logger.warning(f"No h5 files found for split '{split}' in {self.repo_id}")
            return str(self.local_dir)

        # derive npy index files from h5 files (e.g., file.h5 -> file_points.npy)
        npy_files = [f.replace(".h5", "_points.npy") for f in h5_files]

        # download h5 files first
        logger.info(f"Downloading {len(h5_files)} h5 files for split '{split}'...")
        self.download_files(h5_files)

        # try to download npy index files (may not exist for all h5 files)
        logger.info("Downloading index files...")
        for npy_file in npy_files:
            try:
                self.download_file(npy_file)
            except Exception:
                logger.debug(f"Index file not found: {npy_file}")

        return str(self.local_dir)

    def download_all(self, allow_patterns: list[str] | None = None) -> str:
        """Download entire dataset (or files matching patterns).

        By default downloads h5 files and associated *_points.npy index files.
        The npy files contain pre-computed point counts for faster dataset initialization.

        Returns the local directory path.
        """
        if allow_patterns is None:
            allow_patterns = ["*.h5", "*_points.npy"]
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_dir=str(self.local_dir),
            allow_patterns=allow_patterns,
        )
        return str(self.local_dir)

    def get_data_root(self, split: str | list[str] | None = None) -> str:
        """Get local data root, downloading if necessary.

        Downloads both h5 files and *_points.npy index files.
        The npy files contain pre-computed point counts to avoid
        expensive index building on every dataset initialization.

        If split is provided, only downloads files for that split.
        Otherwise downloads all data files.
        """
        if split:
            splits = [split] if isinstance(split, str) else split
            for s in splits:
                self.download_split(s)
        else:
            self.download_all()
        return str(self.local_dir)


def download_pilarnet(
    split: str | list[str] | None = "train",
    repo_id: str | None = None,
    revision: str | None = "main",
    local_dir: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Download PILArNet dataset from HuggingFace.

    Args:
        split: Dataset split(s) to download ("train", "val", "test", or list).
               If None, downloads all data.
        repo_id: HuggingFace repository ID. Defaults to "deeplearnphysics/pilarnet-m".
        revision: Git revision (branch/tag/commit). Defaults to "main".
        local_dir: Local directory for downloaded files.
        token: HuggingFace access token.

    Returns:
        Local directory path containing downloaded files.

    Example:
        >>> data_root = download_pilarnet(split="train")
        >>> dataset = PILArNetH5Dataset(data_root=data_root, split="train")
    """
    interface = PILArNetHFInterface(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=local_dir,
    )
    return interface.get_data_root(split=split)
