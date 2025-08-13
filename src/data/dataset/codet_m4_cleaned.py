import logging
import os
from typing import Tuple, Union, List
from datasets import Dataset, load_from_disk

from datasets import ClassLabel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoDeTM4Cleaned:
    """Dataset class for loading and processing the cleaned CoDet-M4 dataset (duplicates removed)."""

    # Dataset column names - available even before loading
    COLUMN_NAMES = [
        'code', 'target', 'model', 'language', 'source', 'features', 'cleaned_code', 'split'
    ]

    def __init__(self, cleaned_data_path: str):
        """
        Initialize the CoDeTM4Cleaned dataset class.

        Args:
            cleaned_data_path (str): Path to the directory containing cleaned datasets.
                                   Should contain subdirectories: train/, val/, test/

        Raises:
            ValueError: If cleaned_data_path is invalid or does not exist.
            FileNotFoundError: If required split directories are not found.
        """
        if not os.path.isdir(cleaned_data_path):
            raise ValueError(
                f"Cleaned data path '{cleaned_data_path}' does not exist or is not a directory"
            )
        
        self.cleaned_data_path = cleaned_data_path
        
        # Check that required split directories exist
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            split_path = os.path.join(cleaned_data_path, split)
            if not os.path.isdir(split_path):
                raise FileNotFoundError(
                    f"Required split directory '{split_path}' not found. "
                    f"Expected structure: {cleaned_data_path}/{{train,val,test}}/"
                )
        
        # Cache for loaded datasets
        self._train = None
        self._val = None
        self._test = None
        
        logger.info(f"Initialized CoDeTM4Cleaned with cleaned data from: {cleaned_data_path}")

    @staticmethod
    def get_column_names() -> List[str]:
        """
        Get the list of available column names in the CoDet-M4 dataset.

        Returns:
            List[str]: List of column names.
        """
        return CoDeTM4Cleaned.COLUMN_NAMES.copy()

    def _load_split(self, split_name: str) -> Dataset:
        """
        Load a specific split from disk.

        Args:
            split_name (str): Name of the split to load ('train', 'val', 'test').

        Returns:
            Dataset: The loaded dataset split.

        Raises:
            ValueError: If split_name is invalid.
            RuntimeError: If dataset loading fails.
        """
        if split_name not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split_name}'. Must be one of ['train', 'val', 'test']")
        
        try:
            split_path = os.path.join(self.cleaned_data_path, split_name)
            dataset = load_from_disk(split_path)
            
            # Ensure target_binary column exists with proper type
            if 'target_binary' not in dataset.column_names:
                logger.warning(f"target_binary column not found in {split_name} split. Adding it...")
                def map_target_binary(example):
                    target_binary = 0 if example["target"] == "human" else 1
                    return {"target_binary": target_binary}
                
                dataset = dataset.map(map_target_binary, num_proc=8)
                dataset = dataset.cast_column("target_binary", ClassLabel(names=["human", "ai"]))
            
            logger.info(f"Loaded {split_name} split: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {split_name} split from {split_path}: {str(e)}")

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
        dynamic_split_sizing: bool = False,  # Default to False since data is already cleaned
        max_split_ratio: float = 0.2,
        val_ratio: float = None,
        test_ratio: float = None
    ) -> Union[Dataset, Tuple[Dataset, ...]]:
        """
        Load and process the cleaned CoDet-M4 dataset.

        Args:
            split (Union[str, List[str]]): Dataset split(s) to load ('train', 'val', 'test', 'all', or list of splits).
            columns (Union[str, List[str]]): Columns to include ('all' or list of column names).
            train_subset (float): Fraction of training data to load (0.0 to 1.0).
            dynamic_split_sizing (bool): Whether to dynamically limit val/test sizes based on train size.
                                        Default False since cleaned data is already properly sized.
            max_split_ratio (float): Maximum ratio of val/test size to train size when dynamic_split_sizing=True.
            val_ratio (float, optional): Specific ratio for validation set size relative to full validation set.
            test_ratio (float, optional): Specific ratio for test set size relative to full test set.

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
            splits_to_load = [split] if split != 'all' else ['train', 'val', 'test']
        else:
            # split is a list - we need to return a tuple
            return_tuple = True
            invalid_splits = [s for s in split if s not in ['train', 'val', 'test']]
            if invalid_splits:
                raise ValueError(f"Invalid splits {invalid_splits}. Must be from ['train', 'val', 'test']")
            splits_to_load = split

        # Load datasets
        datasets = {}
        
        for split_name in splits_to_load:
            # Use cached dataset if available
            if split_name == 'train' and self._train is not None:
                datasets[split_name] = self._train
            elif split_name == 'val' and self._val is not None:
                datasets[split_name] = self._val
            elif split_name == 'test' and self._test is not None:
                datasets[split_name] = self._test
            else:
                # Load and cache dataset
                dataset = self._load_split(split_name)
                datasets[split_name] = dataset
                
                # Cache the dataset
                if split_name == 'train':
                    self._train = dataset
                elif split_name == 'val':
                    self._val = dataset
                elif split_name == 'test':
                    self._test = dataset

        # Apply train subset if needed
        if 'train' in datasets and train_subset < 1.0:
            datasets['train'] = self._get_train_subset(datasets['train'], train_subset)
            # Update cache
            self._train = datasets['train']

        # Apply dynamic sizing if requested
        if dynamic_split_sizing and 'train' in datasets:
            train_size = len(datasets['train'])
            
            for split_name in ['val', 'test']:
                if split_name in datasets:
                    original_size = len(datasets[split_name])
                    
                    # Calculate target size
                    if split_name == 'val' and val_ratio is not None:
                        target_size = int(original_size * val_ratio)
                        logger.info(f"Target validation size: {target_size} ({val_ratio:.1%} of available validation data)")
                    elif split_name == 'test' and test_ratio is not None:
                        target_size = int(original_size * test_ratio)
                        logger.info(f"Target test size: {target_size} ({test_ratio:.1%} of available test data)")
                    else:
                        target_size = int(train_size * max_split_ratio)
                    
                    if target_size < original_size:
                        datasets[split_name] = self._limit_split_size(datasets[split_name], target_size)
                        logger.info(f"Limited {split_name} from {original_size} to {len(datasets[split_name])} samples")

        # Filter columns for all datasets
        for split_name in datasets:
            datasets[split_name] = self._filter_columns(datasets[split_name], columns)

        # Return results
        if return_tuple:
            return tuple(datasets[split_name] for split_name in splits_to_load)
        elif len(splits_to_load) == 1:
            return datasets[splits_to_load[0]]
        else:
            # Concatenate all splits for 'all' case
            from datasets import concatenate_datasets
            return concatenate_datasets([datasets[split_name] for split_name in splits_to_load])

    def get_info(self) -> dict:
        """
        Get information about the cleaned dataset.

        Returns:
            dict: Dictionary containing dataset information.
        """
        info = {
            'cleaned_data_path': self.cleaned_data_path,
            'available_splits': ['train', 'val', 'test'],
            'column_names': self.get_column_names()
        }
        
        # Add size information if datasets are cached
        if self._train is not None:
            info['train_size'] = len(self._train)
        if self._val is not None:
            info['val_size'] = len(self._val)
        if self._test is not None:
            info['test_size'] = len(self._test)
        
        # Load metadata if available
        metadata_path = os.path.join(self.cleaned_data_path, 'cleaning_metadata.json')
        if os.path.exists(metadata_path):
            import json
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                info['cleaning_metadata'] = metadata
            except Exception as e:
                logger.warning(f"Could not load cleaning metadata: {e}")
        
        return info


if __name__ == "__main__":
    # Example usage - you'll need to update the path to your cleaned data
    CLEANED_DATA_PATH = "../../data/codet_cleaned_20250812_143022/"  # Update this path
    
    if os.path.exists(CLEANED_DATA_PATH):
        # Initialize dataset loader
        dataset_loader = CoDeTM4Cleaned(cleaned_data_path=CLEANED_DATA_PATH)
        
        # Test column names access
        logger.info(f"Available columns: {CoDeTM4Cleaned.get_column_names()}")
        
        # Get dataset info
        info = dataset_loader.get_info()
        logger.info(f"Dataset info: {info}")
        
        # Load single split
        logger.info("Loading training data...")
        train = dataset_loader.get_dataset(split='train')
        logger.info(f"Train dataset size: {len(train)}")
        
        # Load multiple splits (same API as original CoDeTM4)
        logger.info("Loading train, val, and test splits...")
        train, val, test = dataset_loader.get_dataset(split=['train', 'val', 'test'])
        logger.info(f"Train dataset size: {len(train)}")
        logger.info(f"Val dataset size: {len(val)}")
        logger.info(f"Test dataset size: {len(test)}")
        
        # Load with train subset
        logger.info("Loading 10% of training data...")
        train_small = dataset_loader.get_dataset(split='train', train_subset=0.1)
        logger.info(f"Small train dataset size: {len(train_small)}")
        
        # Load specific columns
        logger.info("Loading specific columns...")
        code_only = dataset_loader.get_dataset(split='train', columns=['code', 'target'])
        logger.info(f"Code-only dataset columns: {code_only.column_names}")
        
        # Show sample
        if len(train) > 0:
            logger.info(f"Sample from dataset: {list(train[0].keys())}")
    else:
        logger.error(f"Cleaned data path not found: {CLEANED_DATA_PATH}")
        logger.info("Please run the data leakage analysis notebook first to generate cleaned data.")
