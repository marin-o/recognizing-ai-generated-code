import logging
import os
from typing import Tuple, Union
from datasets import load_dataset, Dataset

from datasets import ClassLabel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AIGCodeSet:
    """Dataset class for loading and processing the AIGCodeSet dataset."""

    def __init__(self, cache_dir: str = "data/"):
        """
        Initialize the AIGCodeSet dataset class.

        Args:
            cache_dir (str): Directory to cache the dataset.

        Raises:
            ValueError: If cache_dir is invalid or does not exist.
        """
        if not os.path.isdir(cache_dir):
            raise ValueError(
                f"Cache directory '{cache_dir}' does not exist or is not a directory"
            )
        self.cache_dir = cache_dir

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset by selecting columns, renaming, and mapping labels.

        Args:
            dataset (Dataset): The raw dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset.
        """
        dataset = dataset.select_columns(["code", "LLM"])
        dataset = dataset.rename_column("LLM", "target")
        dataset = dataset.map(
            lambda x: {"target": "ai" if x["target"] != "Human" else "human"}
        )
        label_map = {"human": 0, "ai": 1}
        dataset = dataset.map(lambda x: {"target": label_map[x["target"]]})
        dataset = dataset.cast_column("target", ClassLabel(names=["human", "ai"]))
        return dataset

    def _split_dataset(
        self, dataset: Dataset, test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split the dataset into train, validation, and test sets.

        Args:
            dataset (Dataset): The dataset to split.
            test_size (float): Proportion of the dataset for the test set.
            val_size (float): Proportion of the training set for the validation set.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.

        Raises:
            ValueError: If split sizes are invalid.
        """
        if not (0 < test_size < 1 and 0 < val_size < 1):
            raise ValueError("test_size and val_size must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")

        train_ds = dataset.train_test_split(
            test_size=test_size, seed=42, stratify_by_column="target"
        )
        train_val = train_ds["train"].train_test_split(
            test_size=val_size / (1 - test_size), seed=42, stratify_by_column="target"
        )
        train = train_val["train"]
        val = train_val["test"]
        test = train_ds["test"]
        return train, val, test

    def get_dataset(
        self, split: bool = True, test_size: float = 0.2, val_size: float = 0.1
    ) -> Union[Dataset, Tuple[Dataset, Dataset, Dataset]]:
        """
        Load and process the AIGCodeSet dataset.

        Args:
            split (bool): Whether to split the dataset into train, val, and test sets.
            test_size (float): Proportion for the test set (if split=True).
            val_size (float): Proportion for the validation set (if split=True).

        Returns:
            Union[Dataset, Tuple[Dataset, Dataset, Dataset]]: Full dataset or (train, val, test) splits.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        try:
            ds = load_dataset(
                "basakdemirok/AIGCodeSet", cache_dir=self.cache_dir, split="train"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

        ds = self._preprocess_dataset(ds)
        if split:
            return self._split_dataset(ds, test_size, val_size)
        return ds


if __name__ == "__main__":
    dataset = AIGCodeSet(cache_dir="data/")
    train, val, test = dataset.get_dataset(split=True)
    logger.info(f"Train dataset size: {len(train)}")
    logger.info(f"Validation dataset size: {len(val)}")
    logger.info(f"Test dataset size: {len(test)}")
    logger.info(f"Sample from train dataset: {train[0]}")
    full_dataset = dataset.get_dataset(split=False)
    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Sample from full dataset: {full_dataset[0]}")
