import logging
import os
from typing import Tuple, Union, List
from datasets import load_dataset, Dataset

from datasets import ClassLabel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoDeTM4:
    """Dataset class for loading and processing the CoDet-M4 dataset."""

    # Dataset column names - available even before loading
    COLUMN_NAMES = [
        'code', 'target', 'model', 'language', 'source', 'features', 'cleaned_code', 'split'
    ]

    def __init__(self, cache_dir: str = "data/"):
        """
        Initialize the CoDeTM4 dataset class.

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

    @staticmethod
    def get_column_names() -> List[str]:
        """
        Get the list of available column names in the CoDet-M4 dataset.

        Returns:
            List[str]: List of column names.
        """
        return CoDeTM4.COLUMN_NAMES.copy()

    def _add_binary_target(self, dataset: Dataset) -> Dataset:
        """
        Add a binary target column to the dataset.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            Dataset: Dataset with added target_binary column.
        """
        # Create binary target mapping
        def map_target_binary(example):
            target_binary = 0 if example["target"] == "human" else 1
            return {"target_binary": target_binary}
        
        dataset = dataset.map(map_target_binary, num_proc=8)
        dataset = dataset.cast_column("target_binary", ClassLabel(names=["human", "ai"]))
        return dataset

    def _filter_columns(self, dataset: Dataset, columns: Union[str, List[str]]) -> Dataset:
        """
        Filter dataset columns based on the columns parameter.

        Args:
            dataset (Dataset): The dataset to filter.
            columns (Union[str, List[str]]): Column specification.

        Returns:
            Dataset: Dataset with filtered columns.

        Raises:
            ValueError: If specified columns are not found in the dataset.
        """
        if columns == 'all':
            return dataset
        
        if isinstance(columns, str):
            columns = [columns]
        
        # Validate columns exist in dataset
        available_columns = list(dataset.column_names)
        invalid_columns = [col for col in columns if col not in available_columns]
        if invalid_columns:
            raise ValueError(f"Columns {invalid_columns} not found in dataset. Available columns: {available_columns}")
        
        # Always include target_binary if it exists
        if 'target_binary' in available_columns and 'target_binary' not in columns:
            columns = columns + ['target_binary']
        
        return dataset.select_columns(columns)

    def _get_train_subset(self, dataset: Dataset, subset_fraction: float) -> Dataset:
        """
        Get a stratified subset of the training dataset to maintain class balance.

        Args:
            dataset (Dataset): The full training dataset.
            subset_fraction (float): Fraction of data to keep (0.0 to 1.0).

        Returns:
            Dataset: Stratified subset of the training dataset.

        Raises:
            ValueError: If subset_fraction is not between 0 and 1.
        """
        if not 0 < subset_fraction <= 1.0:
            raise ValueError("train_subset must be between 0 and 1.0")
        
        if subset_fraction == 1.0:
            return dataset
        
        # Use stratified sampling based on target_binary to maintain class balance
        try:
            subset_data = dataset.train_test_split(
                train_size=subset_fraction,
                seed=42,
                stratify_by_column='target_binary'
            )
            return subset_data['train']
        except Exception as e:
            # Fallback to regular sampling if stratified sampling fails
            logger.warning(f"Stratified sampling failed: {e}. Using regular sampling.")
            total_samples = len(dataset)
            subset_size = int(total_samples * subset_fraction)
            return dataset.select(range(subset_size))

    def _limit_split_size(self, dataset: Dataset, max_size: int) -> Dataset:
        """
        Limit the size of a dataset split using stratified sampling.

        Args:
            dataset (Dataset): The dataset to limit.
            max_size (int): Maximum number of samples to keep.

        Returns:
            Dataset: Limited dataset with stratified sampling if possible.
        """
        if len(dataset) <= max_size:
            return dataset
        
        # Calculate fraction to keep
        fraction = max_size / len(dataset)
        
        # Use stratified sampling if target_binary exists
        try:
            if 'target_binary' in dataset.column_names:
                limited_data = dataset.train_test_split(
                    train_size=fraction,
                    seed=42,
                    stratify_by_column='target_binary'
                )
                return limited_data['train']
            else:
                # Fallback to regular sampling
                return dataset.select(range(max_size))
        except Exception as e:
            logger.warning(f"Stratified sampling failed for split limiting: {e}. Using regular sampling.")
            return dataset.select(range(max_size))

    def get_dataset(
        self, 
        split: Union[str, List[str]] = 'all', 
        columns: Union[str, List[str]] = 'all', 
        train_subset: float = 1.0,
        dynamic_split_sizing: bool = True,
        max_split_ratio: float = 0.2,
        val_ratio: float = None,
        test_ratio: float = None
    ) -> Union[Dataset, Tuple[Dataset, ...]]:
        """
        Load and process the CoDet-M4 dataset.

        Args:
            split (Union[str, List[str]]): Dataset split(s) to load ('train', 'val', 'test', 'all', or list of splits).
            columns (Union[str, List[str]]): Columns to include ('all' or list of column names).
            train_subset (float): Fraction of training data to load (0.0 to 1.0).
            dynamic_split_sizing (bool): Whether to dynamically limit val/test sizes based on train size.
            max_split_ratio (float): Maximum ratio of val/test size to train size when dynamic_split_sizing=True.
                                   Used only if val_ratio/test_ratio are not specified (backward compatibility).
            val_ratio (float, optional): Specific ratio for validation set size relative to total dataset.
                                       If specified, overrides max_split_ratio for validation set.
            test_ratio (float, optional): Specific ratio for test set size relative to total dataset.
                                        If specified, overrides max_split_ratio for test set.

        Returns:
            Union[Dataset, Tuple[Dataset, ...]]: Single dataset if split is string, tuple of datasets if split is list.

        Raises:
            ValueError: If split is invalid or columns are not found.
            RuntimeError: If dataset loading fails.
        """
        # Validate split parameter
        valid_splits = ['train', 'val', 'test', 'all']
        
        # Handle split parameter - convert to list for unified processing
        return_tuple = False
        if isinstance(split, str):
            if split not in valid_splits:
                raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
            splits_to_load = [split] if split != 'all' else valid_splits[:-1]  # exclude 'all' itself
        else:
            # split is a list - we need to return a tuple
            return_tuple = True
            invalid_splits = [s for s in split if s not in valid_splits[:-1]]  # exclude 'all' from valid list splits
            if invalid_splits:
                raise ValueError(f"Invalid splits {invalid_splits}. Must be from {valid_splits[:-1]}")
            splits_to_load = split

        try:
            # Load the entire dataset (splits are indicated by 'split' column)
            ds = load_dataset(
                "DaniilOr/CoDET-M4", 
                cache_dir=self.cache_dir
            )
            # If dataset has multiple splits, concatenate them
            if isinstance(ds, dict):
                from datasets import concatenate_datasets
                all_datasets = []
                for split_name in ds.keys():
                    all_datasets.append(ds[split_name])
                ds = concatenate_datasets(all_datasets)
            else:
                # If dataset is already a single Dataset object
                ds = ds
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

        # Add binary target column
        ds = self._add_binary_target(ds)

        # If we need to return separate splits, process each split individually
        if return_tuple:
            split_datasets = []
            train_size = None
            
            # First pass: get train size if train is requested
            if 'train' in splits_to_load and dynamic_split_sizing:
                train_ds = ds.filter(lambda x: x['split'] == 'train', num_proc=8)
                if train_subset < 1.0:
                    train_ds = self._get_train_subset(train_ds, train_subset)
                train_size = len(train_ds)
                
                # For legacy behavior: limit val/test relative to train size
                if val_ratio is None and test_ratio is None:
                    max_other_split_size = int(train_size * max_split_ratio)
                    logger.info(f"Train size: {train_size}, max val/test size: {max_other_split_size}")
                else:
                    logger.info(f"Train size: {train_size}")
            
            # Second pass: process all splits
            for split_name in splits_to_load:
                # Filter for this specific split
                split_ds = ds.filter(lambda x: x['split'] == split_name, num_proc=8)
                
                # Apply train subset if this is training data
                if split_name == 'train' and train_subset < 1.0:
                    split_ds = self._get_train_subset(split_ds, train_subset)
                
                # Apply dynamic sizing for val/test splits
                elif split_name in ['val', 'test'] and dynamic_split_sizing and train_size is not None:
                    original_size = len(split_ds)
                    
                    # Use new ratio-based sizing if available
                    if split_name == 'val' and val_ratio is not None:
                        # Calculate target size as ratio of the original validation set size
                        target_size = int(original_size * val_ratio)
                        logger.info(f"Target validation size: {target_size} ({val_ratio:.1%} of available validation data)")
                        split_ds = self._limit_split_size(split_ds, target_size)
                    elif split_name == 'test' and test_ratio is not None:
                        # Calculate target size as ratio of the original test set size
                        target_size = int(original_size * test_ratio)
                        logger.info(f"Target test size: {target_size} ({test_ratio:.1%} of available test data)")
                        split_ds = self._limit_split_size(split_ds, target_size)
                    else:
                        # Fallback to legacy behavior
                        max_other_split_size = int(train_size * max_split_ratio)
                        split_ds = self._limit_split_size(split_ds, max_other_split_size)
                    
                    if len(split_ds) < original_size:
                        logger.info(f"Limited {split_name} from {original_size} to {len(split_ds)} samples")
                
                # Filter columns
                split_ds = self._filter_columns(split_ds, columns)
                split_datasets.append(split_ds)
            
            return tuple(split_datasets)
        
        else:
            # Single split or 'all' - return single dataset
            # Filter by split column if not loading all splits
            if isinstance(split, str) and split != 'all':
                ds = ds.filter(lambda x: x['split'] in splits_to_load, num_proc=8)

            # Apply train subset if this includes training data
            if 'train' in splits_to_load and train_subset < 1.0:
                ds = self._get_train_subset(ds, train_subset)

            # Filter columns
            ds = self._filter_columns(ds, columns)
            
            # Dynamic split sizing for validation and test splits
            if dynamic_split_sizing and 'train' in splits_to_load:
                train_size = len(ds)
                
                # Limit validation and test splits if they exceed the maximum size
                if 'val' in splits_to_load:
                    val_split = ds.filter(lambda x: x['split'] == 'val', num_proc=8)
                    original_val_size = len(val_split)
                    
                    # Calculate target size using ratio if provided, otherwise use legacy logic
                    if val_ratio is not None:
                        max_val_size = int(original_val_size * val_ratio)
                        logger.info(f"Target validation size: {max_val_size} ({val_ratio:.1%} of available validation data)")
                    else:
                        max_val_size = int(train_size * max_split_ratio)
                    
                    val_split = self._limit_split_size(val_split, max_val_size)
                    if len(val_split) < original_val_size:
                        logger.info(f"Limited validation from {original_val_size} to {len(val_split)} samples")
                    ds = ds.filter(lambda x: x['split'] != 'val', num_proc=8).concatenate(val_split)
                
                if 'test' in splits_to_load:
                    test_split = ds.filter(lambda x: x['split'] == 'test', num_proc=8)
                    original_test_size = len(test_split)
                    
                    # Calculate target size using ratio if provided, otherwise use legacy logic
                    if test_ratio is not None:
                        max_test_size = int(original_test_size * test_ratio)
                        logger.info(f"Target test size: {max_test_size} ({test_ratio:.1%} of available test data)")
                    else:
                        max_test_size = int(train_size * max_split_ratio)
                    
                    test_split = self._limit_split_size(test_split, max_test_size)
                    if len(test_split) < original_test_size:
                        logger.info(f"Limited test from {original_test_size} to {len(test_split)} samples")
                    ds = ds.filter(lambda x: x['split'] != 'test', num_proc=8).concatenate(test_split)

            return ds


if __name__ == "__main__":
    # Example usage
    dataset_loader = CoDeTM4(cache_dir="data/")
    
    # Test column names access
    logger.info(f"Available columns: {CoDeTM4.get_column_names()}")
    
    # Load full dataset (single dataset)
    logger.info("Loading full dataset...")
    full_dataset = dataset_loader.get_dataset(split='all')
    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Full dataset columns: {full_dataset.column_names}")
    
    # Load single split (single dataset)
    logger.info("Loading 10% of training data...")
    train_subset = dataset_loader.get_dataset(split='train', train_subset=0.1)
    logger.info(f"Train subset size: {len(train_subset)}")
    
    # Load multiple splits with dynamic sizing (tuple of datasets)
    logger.info("Loading train, val, and test splits with dynamic sizing...")
    train, val, test = dataset_loader.get_dataset(
        split=['train', 'val', 'test'], 
        train_subset=0.05,  # 5% of train data
        dynamic_split_sizing=True,
        max_split_ratio=0.2  # val and test will be at most 20% of train size
    )
    logger.info(f"Train dataset size: {len(train)}")
    logger.info(f"Val dataset size: {len(val)}")
    logger.info(f"Test dataset size: {len(test)}")
    
    # Load multiple splits with specific ratios (NEW FUNCTIONALITY)
    logger.info("Loading train, val, and test splits with specific ratios...")
    train2, val2, test2 = dataset_loader.get_dataset(
        split=['train', 'val', 'test'], 
        train_subset=0.05,  # 5% of train data
        dynamic_split_sizing=True,
        val_ratio=0.15,  # 15% of total dataset for validation
        test_ratio=0.25  # 25% of total dataset for test
    )
    logger.info(f"Train dataset size: {len(train2)}")
    logger.info(f"Val dataset size: {len(val2)} (15% of total)")
    logger.info(f"Test dataset size: {len(test2)} (25% of total)")
    
    # Load specific columns
    logger.info("Loading specific columns...")
    code_only = dataset_loader.get_dataset(split='train', columns=['code', 'target'])
    logger.info(f"Code-only dataset columns: {code_only.column_names}")
    
    # Show sample
    if len(full_dataset) > 0:
        logger.info(f"Sample from dataset: {full_dataset[0]}")
