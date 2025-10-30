import logging
import os
from typing import Tuple, Union, List, Optional
from datasets import Dataset, load_dataset

from datasets import ClassLabel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SemEval2026Task13:
    """Dataset class for loading and processing the SemEval 2026 Task 13 dataset from Hugging Face.
    
    Expected columns: code, generator, label, language
    The 'label' column contains the classification target (e.g., 'human' vs 'ai').
    A 'target_binary' column is automatically added for binary classification (0=human, 1=ai).
    """

    # Dataset identifier
    HF_DATASET_NAME = "DaniilOr/SemEval-2026-Task13"

    def __init__(self, subtask: str = "A"):
        """
        Initialize the SemEval2026Task13 class.

        Args:
            subtask (str): Subtask identifier ('A', 'B', or 'C').

        Raises:
            ValueError: If subtask is invalid.
        """
        if subtask not in ['A', 'B', 'C']:
            raise ValueError(f"Invalid subtask '{subtask}'. Must be one of ['A', 'B', 'C']")
        
        self.subtask = subtask
        
        # Cache for loaded datasets
        self._train = None
        self._val = None
        self._test = None
        
        logger.info(f"Initialized SemEval2026Task13 for subtask {subtask}")

    @staticmethod
    def get_available_subtasks() -> List[str]:
        """
        Get the list of available subtasks.

        Returns:
            List[str]: List of available subtasks.
        """
        return ['A', 'B', 'C']

    @staticmethod
    def get_column_names(subtask: str = "A") -> List[str]:
        """
        Get the list of available column names for a specific subtask.
        Note: Column names may vary by subtask and are determined dynamically at runtime.

        Args:
            subtask (str): The subtask to get column names for.

        Returns:
            List[str]: List of column names (empty list until dataset is loaded).
        """
        # Column names will be determined after loading the dataset
        # Use get_info() method after initialization to see actual columns
        return []

    def _load_dataset(self):
        """
        Load the full dataset from Hugging Face.

        Returns:
            The loaded dataset (DatasetDict or Dataset).

        Raises:
            RuntimeError: If dataset loading fails.
        """
        try:
            logger.info(f"Loading SemEval 2026 Task 13 subtask {self.subtask} from Hugging Face...")
            dataset = load_dataset(self.HF_DATASET_NAME, self.subtask)
            logger.info(f"Loaded dataset with splits: {list(dataset.keys()) if hasattr(dataset, 'keys') else 'single dataset'}")  # type: ignore
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load SemEval 2026 Task 13 subtask {self.subtask}: {str(e)}")

    def _load_split(self, split_name: str) -> Dataset:
        """
        Load a specific split from the Hugging Face dataset.

        Args:
            split_name (str): Name of the split to load ('train', 'validation', 'test').

        Returns:
            Dataset: The loaded dataset split.

        Raises:
            ValueError: If split_name is invalid.
            RuntimeError: If dataset loading fails.
        """
        # Map common split names
        split_mapping = {
            'val': 'validation',
            'dev': 'validation'
        }
        hf_split_name = split_mapping.get(split_name, split_name)
        
        try:
            full_dataset = self._load_dataset()
            
            # Handle DatasetDict (multiple splits)
            if hasattr(full_dataset, 'keys') and hf_split_name in full_dataset:
                dataset = full_dataset[hf_split_name]
            elif hasattr(full_dataset, 'keys'):
                available_splits = list(full_dataset.keys())  # type: ignore
                raise ValueError(f"Split '{hf_split_name}' not found in dataset. Available splits: {available_splits}")
            else:
                # Single dataset, return as-is
                dataset = full_dataset
            
            # Ensure target_binary column exists with proper type if label column exists
            if hasattr(dataset, 'column_names') and 'label' in dataset.column_names and 'target_binary' not in dataset.column_names:  # type: ignore
                logger.info(f"Adding target_binary column to {hf_split_name} split...")
                def map_target_binary(example):
                    # Assuming label is 0 (human) or 1 (ai) or similar integer
                    label_value = example.get("label", 0)
                    if isinstance(label_value, str):
                        target_binary = 0 if label_value.lower() == "human" else 1
                    else:
                        # Assume integer: 0 = human, 1 = ai
                        target_binary = int(label_value)
                    return {"target_binary": target_binary}
                
                dataset = dataset.map(map_target_binary)  # type: ignore
                if hasattr(dataset, 'cast_column'):
                    dataset = dataset.cast_column("target_binary", ClassLabel(names=["human", "ai"]))
            
            logger.info(f"Loaded {hf_split_name} split: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples")  # type: ignore
            return dataset  # type: ignore
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {hf_split_name} split: {str(e)}")

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
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
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
        Get information about the SemEval 2026 Task 13 dataset.

        Returns:
            dict: Dictionary containing dataset information.
        """
        info = {
            'dataset_name': self.HF_DATASET_NAME,
            'subtask': self.subtask,
            'available_subtasks': self.get_available_subtasks()
        }
        
        # Add size information if datasets are cached
        if self._train is not None:
            info['train_size'] = len(self._train)
        if self._val is not None:
            info['val_size'] = len(self._val)
        if self._test is not None:
            info['test_size'] = len(self._test)
        
        # Try to load dataset to get more info if not cached
        try:
            dataset = self._load_dataset()
            if hasattr(dataset, 'keys') and callable(getattr(dataset, 'keys', None)):
                info['available_splits'] = list(dataset.keys())  # type: ignore
                # Add size information for each split if not cached
                for split_name in dataset.keys():  # type: ignore
                    if f'{split_name}_size' not in info:  # Don't override cached sizes
                        split_data = dataset[split_name]  # type: ignore
                        if hasattr(split_data, '__len__'):
                            info[f'{split_name}_size'] = len(split_data)  # type: ignore
                        if hasattr(split_data, 'column_names'):
                            info[f'{split_name}_columns'] = split_data.column_names  # type: ignore
            else:
                info['dataset_type'] = 'single_split'
                if 'size' not in info and hasattr(dataset, '__len__'):
                    info['size'] = len(dataset)  # type: ignore
                if hasattr(dataset, 'column_names'):
                    info['columns'] = dataset.column_names
        except Exception as e:
            logger.warning(f"Could not load dataset for info: {e}")
        
        return info


if __name__ == "__main__":
    # Example usage - requires huggingface-cli login for access
    # Login using e.g. `huggingface-cli login` to access this dataset
    
    # Initialize dataset loader for subtask A
    dataset_loader = SemEval2026Task13(subtask="A")
    
    # Test available subtasks
    logger.info(f"Available subtasks: {SemEval2026Task13.get_available_subtasks()}")
    
    # Get dataset info
    info = dataset_loader.get_info()
    logger.info(f"Dataset info: {info}")
    
    try:
        # Load single split
        logger.info("Loading training data...")
        train = dataset_loader.get_dataset(split='train')
        logger.info(f"Train dataset size: {len(train) if hasattr(train, '__len__') else 'unknown'}")
        
        # Load multiple splits (same API as original)
        logger.info("Loading train, val, and test splits...")
        train, val, test = dataset_loader.get_dataset(split=['train', 'val', 'test'])
        logger.info(f"Train dataset size: {len(train) if hasattr(train, '__len__') else 'unknown'}")
        logger.info(f"Val dataset size: {len(val) if hasattr(val, '__len__') else 'unknown'}")
        logger.info(f"Test dataset size: {len(test) if hasattr(test, '__len__') else 'unknown'}")
        
        # Load with train subset
        logger.info("Loading 10% of training data...")
        train_small = dataset_loader.get_dataset(split='train', train_subset=0.1)
        logger.info(f"Small train dataset size: {len(train_small) if hasattr(train_small, '__len__') else 'unknown'}")
        
        # Load specific columns
        logger.info("Loading specific columns...")
        code_only = dataset_loader.get_dataset(split='train', columns=['code', 'label'])
        logger.info(f"Code-only dataset columns: {code_only.column_names if hasattr(code_only, 'column_names') else 'unknown'}")  # type: ignore
        
        # Show sample
        if hasattr(train, '__len__') and len(train) > 0:
            logger.info(f"Sample from dataset: {list(train[0].keys()) if hasattr(train, '__getitem__') and hasattr(train[0], 'keys') else 'sample available'}")
    
    except Exception as e:
        logger.error(f"Error loading dataset. Make sure you have logged in with 'huggingface-cli login': {e}")
        logger.info("To login: huggingface-cli login")
