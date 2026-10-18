import importlib.util
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from ._validation import validate_partitioner_int
from .dirichlet import dirichlet_partition
from .hybrid import hybrid_partition
from .iid import iid_partition
from .pathological import pathological_partition
from .quantity import quantity_skew_partition

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None

Dataset = object
DataLoader = None
Subset = None
tv_datasets = None


class NumpyDataset(Dataset):
    """Lightweight dataset wrapper for NumPy arrays or tensor tuples.

    Parameters
    ----------
    data : Sequence or np.ndarray or torch.Tensor
        Feature array or tensor.
    targets : Sequence or np.ndarray or torch.Tensor
        Target labels array or tensor.
    transform : callable, optional
        Transformation applied to feature samples.
    target_transform : callable, optional
        Transformation applied to target labels.
    """

    def __init__(
        self,
        data: Any,
        targets: Any,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class FederatedDataset:
    """High-level Federated Dataset manager for simulated distributed learning.

    Provides automatic downloading, loading, and partitioning of datasets across
    simulated clients according to standard Federated Learning benchmark distributions.

    Supported Partitioning Regimes
    ------------------------------
    - **"iid"**:
      Uniform random data splitting across all clients. Every client receives roughly
      equal sample quantities and an identical class distribution.

    - **"dirichlet"** (Label Skew):
      Continuous non-IID label distribution based on Dirichlet sampling Dir(alpha).
      * **alpha -> infinity** (e.g., 100.0): Approaches uniform, near-IID class distribution.
      * **alpha ≈ 0.5**: Standard benchmark setting representing realistic heterogeneous clients.
      * **alpha -> 0** (e.g., 0.05, 0.01): Extreme non-IID label skew where each client predominantly
        holds samples from only 1 or 2 classes.

    - **"pathological"** (Shard-based Label Skew):
      The classic FedAvg benchmark (McMahan et al., 2017). Samples are sorted by label,
      split into shards, and each client is assigned shards from a fixed number of classes
      (configured via `classes_per_client`, e.g., 2 classes per client for MNIST).

    - **"quantity"** (Sample Volume Skew):
      Simulates hardware/usage heterogeneity where different clients collect vastly different
      volumes of data. Client sample sizes follow a Dirichlet distribution parameterized by `beta`.
      * **beta -> infinity** (e.g., 100.0): Clients receive roughly equal sample volumes.
      * **beta -> 0** (e.g., 0.1): Extreme volume disparity (some clients receive hundreds of samples,
        while others receive single digits).

    - **"hybrid"** (Joint Label & Quantity Skew):
      Combines volume disparity (governed by `beta`) with label preference heterogeneity
      (governed by `alpha`).

    Parameters
    ----------
    dataset : str or Tuple[Any, Any] or Dataset
        Dataset identifier or instance:
        - String name for built-in torchvision datasets: `"mnist"`, `"cifar10"`,
          `"cifar100"`, `"fashion_mnist"`.
        - Tuple of `(X, y)` containing features and labels (NumPy arrays or lists).
        - Existing PyTorch `Dataset` object.
    data_dir : str, default="./data"
        Directory path used to cache and download standard datasets.
    num_clients : int
        Number of participating federated clients.
    partition : str, default="iid"
        Partition strategy. Choices: `"iid"`, `"dirichlet"`, `"pathological"`,
        `"quantity"`, `"hybrid"`.
    alpha : float, default=0.5
        Dirichlet concentration parameter for label skew. Smaller values yield stronger
        label heterogeneity across clients.
    beta : float, default=0.5
        Dirichlet concentration parameter for quantity skew. Smaller values yield higher
        variance in client sample counts.
    classes_per_client : int, default=2
        Number of distinct classes assigned to each client in `"pathological"` partition.
    seed : int, optional
        Random seed for reproducible dataset partitioning.
    train : bool, default=True
        Whether to load the training or test split when downloading built-in datasets.
    download : bool, default=True
        Whether to automatically download built-in torchvision datasets if not present locally.
    transform : callable, optional
        Feature transformation applied to samples.
    target_transform : callable, optional
        Target transformation applied to labels.

    Examples
    --------
    >>> # Using standard MNIST with Dirichlet label skew:
    >>> fed_data = FederatedDataset("mnist", num_clients=10, partition="dirichlet", alpha=0.5)
    >>> client_0_loader = fed_data.get_client_dataloader(client_id=0, batch_size=32)

    >>> # Using custom NumPy arrays with quantity skew:
    >>> import numpy as np
    >>> X = np.random.randn(1000, 20)
    >>> y = np.random.randint(0, 5, size=1000)
    >>> fed_data = FederatedDataset((X, y), num_clients=5, partition="quantity", beta=0.2)
    >>> indices = fed_data.get_client_indices(client_id=0)
    """

    SUPPORTED_DATASETS = {
        "mnist": "MNIST",
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
        "fashion_mnist": "FashionMNIST",
        "fashionmnist": "FashionMNIST",
    }

    SUPPORTED_PARTITIONS = {
        "iid",
        "dirichlet",
        "pathological",
        "quantity",
        "hybrid",
    }

    def __init__(
        self,
        dataset: Union[str, Tuple[Any, Any], Any],
        data_dir: str = "./data",
        num_clients: int = 10,
        partition: str = "iid",
        alpha: float = 0.5,
        beta: float = 0.5,
        classes_per_client: int = 2,
        seed: Optional[int] = None,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.partition = partition.lower()
        self.alpha = alpha
        self.beta = beta
        self.classes_per_client = classes_per_client
        self.seed = seed
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        self.num_clients = validate_partitioner_int("num_clients", num_clients, min_value=1)
        if self.partition not in self.SUPPORTED_PARTITIONS:
            raise ValueError(
                f"Unsupported partition '{self.partition}'. "
                f"Choose from: {sorted(self.SUPPORTED_PARTITIONS)}"
            )
        if self.partition == "pathological":
            self.classes_per_client = validate_partitioner_int(
                "classes_per_client", classes_per_client, min_value=1
            )

        # 1. Load the underlying base dataset and extract target labels
        self.base_dataset, self.labels = self._load_dataset(dataset)

        # 2. Perform the specified partitioning
        self.client_dict = self._partition_data()

    def _load_dataset(self, dataset: Any) -> Tuple[Any, np.ndarray]:
        """Resolve dataset input into a dataset instance and NumPy labels array."""
        if isinstance(dataset, str):
            ds_key = dataset.lower().strip()
            if ds_key not in self.SUPPORTED_DATASETS:
                raise ValueError(
                    f"Unknown built-in dataset '{dataset}'. "
                    f"Supported names: {sorted(self.SUPPORTED_DATASETS.keys())}"
                )
            if not _HAS_TORCHVISION:
                raise ImportError(
                    f"torchvision is required to download and load '{dataset}'. "
                    f"Please install torchvision or provide custom (X, y) data."
                )

            global tv_datasets
            if tv_datasets is None:
                try:
                    import torchvision.datasets as _tv_datasets

                    tv_datasets = _tv_datasets
                except ImportError:
                    raise ImportError(
                        f"torchvision is required to download and load '{dataset}'. "
                        f"Please install torchvision or provide custom (X, y) data."
                    )

            ds_class_name = self.SUPPORTED_DATASETS[ds_key]
            ds_cls = getattr(tv_datasets, ds_class_name)

            ds_instance = ds_cls(
                root=self.data_dir,
                train=self.train,
                download=self.download,
                transform=self.transform,
                target_transform=self.target_transform,
            )

            # Extract targets
            if hasattr(ds_instance, "targets"):
                targets = ds_instance.targets
                if hasattr(targets, "numpy"):
                    labels = targets.numpy()
                else:
                    labels = np.asarray(targets)
            else:
                labels = np.array([ds_instance[i][1] for i in range(len(ds_instance))])

            return ds_instance, labels

        if isinstance(dataset, tuple) and len(dataset) == 2:
            data, targets = dataset
            if len(data) != len(targets):
                raise ValueError(
                    f"Mismatch between features length ({len(data)}) and "
                    f"targets length ({len(targets)})"
                )
            labels = np.asarray(targets)
            ds_instance = NumpyDataset(
                data=data,
                targets=targets,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            return ds_instance, labels

        # Generic dataset object (e.g. custom PyTorch Dataset)
        if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            if hasattr(dataset, "targets"):
                targets = dataset.targets
                labels = targets.numpy() if hasattr(targets, "numpy") else np.asarray(targets)
            elif hasattr(dataset, "labels"):
                labels = np.asarray(dataset.labels)
            else:
                labels = np.array([dataset[i][1] for i in range(len(dataset))])
            return dataset, labels

        raise TypeError(
            f"dataset must be a string name, (X, y) tuple, or Dataset instance. Got: {type(dataset)}"
        )

    def _partition_data(self) -> Dict[int, np.ndarray]:
        """Dispatch partitioning according to configured strategy."""
        if self.partition == "iid":
            return iid_partition(
                labels=self.labels,
                num_clients=self.num_clients,
                seed=self.seed,
            )
        elif self.partition == "dirichlet":
            return dirichlet_partition(
                labels=self.labels,
                num_clients=self.num_clients,
                alpha=self.alpha,
                seed=self.seed,
            )
        elif self.partition == "pathological":
            return pathological_partition(
                labels=self.labels,
                num_clients=self.num_clients,
                classes_per_client=self.classes_per_client,
                seed=self.seed,
            )
        elif self.partition == "quantity":
            return quantity_skew_partition(
                labels=self.labels,
                num_clients=self.num_clients,
                beta=self.beta,
                seed=self.seed,
            )
        elif self.partition == "hybrid":
            return hybrid_partition(
                labels=self.labels,
                num_clients=self.num_clients,
                alpha=self.alpha,
                beta=self.beta,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition}")

    def get_client_indices(self, client_id: int) -> np.ndarray:
        """Get the array of sample indices allocated to a specific client.

        Parameters
        ----------
        client_id : int
            Index of the client (0 <= client_id < num_clients).

        Returns
        -------
        np.ndarray
            1D array of sample indices.
        """
        if client_id not in self.client_dict:
            raise IndexError(
                f"Client ID {client_id} out of bounds for num_clients={self.num_clients}"
            )
        return self.client_dict[client_id]

    def get_client_dataset(self, client_id: int) -> Any:
        """Get a Dataset instance containing only the client's assigned samples.

        Parameters
        ----------
        client_id : int
            Index of the client.

        Returns
        -------
        Dataset or Subset
            Client dataset wrapper.
        """
        indices = self.get_client_indices(client_id)
        if _HAS_TORCH:
            global Subset
            if Subset is None:
                try:
                    from torch.utils.data import Subset as _Subset

                    Subset = _Subset
                except ImportError:
                    Subset = None
            if Subset is not None:
                return Subset(self.base_dataset, indices)

        # Fallback subset wrapper for non-torch environments
        class _NonTorchSubset:
            def __init__(self, base, idxs):
                self.base = base
                self.indices = idxs

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                return self.base[self.indices[i]]

        return _NonTorchSubset(self.base_dataset, indices)

    def get_client_dataloader(
        self,
        client_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Get a PyTorch DataLoader for the specified client.

        Parameters
        ----------
        client_id : int
            Index of the client.
        batch_size : int, default=32
            Mini-batch size for client local training.
        shuffle : bool, default=True
            Whether to shuffle the client's data each epoch.
        **kwargs : Any
            Additional keyword arguments passed to `torch.utils.data.DataLoader`.

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader for local client training.
        """
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required to create a DataLoader. "
                "Install torch (`pip install torch`) or use get_client_dataset() / get_client_indices()."
            )

        global DataLoader
        if DataLoader is None:
            try:
                from torch.utils.data import DataLoader as _DataLoader

                DataLoader = _DataLoader
            except ImportError:
                raise ImportError(
                    "PyTorch is required to create a DataLoader. "
                    "Install torch (`pip install torch`) or use get_client_dataset() / get_client_indices()."
                )

        client_ds = self.get_client_dataset(client_id)
        return DataLoader(client_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

    @property
    def client_sizes(self) -> Dict[int, int]:
        """Dictionary mapping client IDs to assigned sample counts."""
        return {cid: len(indices) for cid, indices in self.client_dict.items()}

    def __len__(self) -> int:
        """Total number of samples in the partitioned dataset."""
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"FederatedDataset(num_clients={self.num_clients}, "
            f"partition='{self.partition}', total_samples={len(self)})"
        )
