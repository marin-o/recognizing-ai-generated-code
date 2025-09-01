import logging
import os
from typing import Tuple, Union, List, Dict
from datasets import load_dataset, Dataset, ClassLabel
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def traverse(node: Node, depth: int) -> int:
    """
    Traverse the CST and compute the maximum nesting depth.

    Args:
        node (Node): The current node in the CST.
        depth (int): The current depth in the tree.

    Returns:
        int: The maximum nesting depth encountered.
    """
    max_nesting_depth = depth

    node_type = node.type
    if node_type in {
        "function_definition",
        "class_definition",
        "if_statement",
        "for_statement",
        "while_statement",
    }:
        for child in node.children:
            max_nesting_depth = max(max_nesting_depth, traverse(child, depth + 1))
    else:
        for child in node.children:
            max_nesting_depth = max(max_nesting_depth, traverse(child, depth))

    return max_nesting_depth


def extract_features_for_example(example: Dict, as_tensor: bool) -> Dict:
    """
    Extract CST features for a single example.

    Args:
        example (Dict): A single dataset example containing 'code'.
        as_tensor (bool): Whether to return features as a tensor-like list.

    Returns:
        Dict: The example with added CST features.
    """
    # Create new parser and language instance for thread safety
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    try:
        tree = parser.parse(bytes(example["code"], "utf8"))
    except Exception as e:
        logger.warning(f"Failed to parse code: {str(e)}")
        return example  # Return unchanged example on parse failure

    root_node = tree.root_node
    lang = PY_LANGUAGE

    # Individual Tree-Sitter queries for each feature
    function_query = lang.query("(function_definition) @func")
    if_query = lang.query("(if_statement) @if_stmt")
    while_query = lang.query("(while_statement) @while_stmt")
    for_query = lang.query("(for_statement) @for_stmt")
    comment_query = lang.query("(comment) @comment")
    import_query = lang.query("(import_statement) @import")
    import_from_query = lang.query("(import_from_statement) @import_from")
    class_query = lang.query("(class_definition) @class_def")
    binary_op_query = lang.query("(binary_operator) @binop")
    error_query = lang.query("(ERROR) @error")

    max_nesting_depth = traverse(root_node, 0)

    features = {
        "function_defs": len(function_query.captures(root_node)),
        "if_statements": len(if_query.captures(root_node)),
        "loops": len(for_query.captures(root_node))
        + len(while_query.captures(root_node)),
        "imports": len(import_query.captures(root_node))
        + len(import_from_query.captures(root_node)),
        "comments": len(comment_query.captures(root_node)),
        "class_defs": len(class_query.captures(root_node)),
        "max_nesting_depth": max_nesting_depth,
        "binary_ops": len(binary_op_query.captures(root_node)),
        "errors": len(error_query.captures(root_node)),
    }
    if as_tensor:
        features = {"cst_features": list(features.values())}
    example.update(features)
    return example


class CoDeTM4_WithCSTFeatures:
    """Dataset class for loading and processing the CoDet-M4 dataset with CST features."""

    # Dataset column names - available even before loading
    COLUMN_NAMES = [
        'code', 'target', 'model', 'language', 'source', 'features', 'cleaned_code', 'split'
    ]

    def __init__(self, cache_dir: str = "data/", features_as_tensor: bool = True):
        """
        Initialize the CoDeTM4_WithCSTFeatures dataset class.

        Args:
            cache_dir (str): Directory to cache the dataset.
            features_as_tensor (bool): Whether to return CST features as tensor-like list.

        Raises:
            ValueError: If cache_dir is invalid or does not exist.
        """
        if not os.path.isdir(cache_dir):
            raise ValueError(
                f"Cache directory '{cache_dir}' does not exist or is not a directory"
            )
        self.cache_dir = cache_dir
        self.features_as_tensor = features_as_tensor

    @staticmethod
    def get_column_names() -> List[str]:
        """
        Get the list of available column names in the CoDet-M4 dataset.

        Returns:
            List[str]: List of column names.
        """
        return CoDeTM4_WithCSTFeatures.COLUMN_NAMES.copy()

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

    def _extract_cst_features(self, data: Dataset) -> Dataset:
        """
        Extract CST features for all examples in the dataset.

        Args:
            data (Dataset): The dataset to process.

        Returns:
            Dataset: The dataset with added CST features.
        """
        return data.map(
            lambda x: extract_features_for_example(
                x, as_tensor=self.features_as_tensor
            ),
            num_proc=12,  # Parallelize across 4 processes
        )

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
            columns.append('target_binary')
        
        return dataset.select_columns(columns)

    def _get_train_subset(self, dataset: Dataset, subset_fraction: float) -> Dataset:
        """
        Get a subset of the training dataset.

        Args:
            dataset (Dataset): The training dataset.
            subset_fraction (float): Fraction of data to keep (0.0 to 1.0).

        Returns:
            Dataset: Subset of the training dataset.

        Raises:
            ValueError: If subset_fraction is not between 0 and 1.
        """
        if not 0 < subset_fraction <= 1:
            raise ValueError("subset_fraction must be between 0 and 1")
        
        num_samples = int(len(dataset) * subset_fraction)
        return dataset.select(range(num_samples))

    def _limit_split_size(self, dataset: Dataset, max_size: int) -> Dataset:
        """
        Limit the size of a dataset split.

        Args:
            dataset (Dataset): The dataset to limit.
            max_size (int): Maximum number of samples.

        Returns:
            Dataset: Limited dataset.
        """
        if len(dataset) > max_size:
            return dataset.select(range(max_size))
        return dataset

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
        Load and process the CoDet-M4 dataset with CST features.

        Args:
            split (Union[str, List[str]]): Which split(s) to load. 
                Options: 'all', 'train', 'val', 'test', or list of these.
            columns (Union[str, List[str]]): Which columns to include. 
                'all' for all columns, or list of specific column names.
            train_subset (float): Fraction of training data to use (0.0 to 1.0).
            dynamic_split_sizing (bool): Whether to dynamically size val/test relative to train.
            max_split_ratio (float): Maximum ratio of val/test size to train size (when dynamic_split_sizing=True).
            val_ratio (float): Specific ratio for validation set size relative to available val data.
            test_ratio (float): Specific ratio for test set size relative to available test data.

        Returns:
            Union[Dataset, Tuple[Dataset, ...]]: Dataset or tuple of datasets based on split parameter.

        Raises:
            RuntimeError: If dataset loading fails.
            ValueError: If split or other parameters are invalid.
        """
        # Normalize split parameter
        if isinstance(split, str):
            if split == 'all':
                splits_to_load = ['train', 'val', 'test']
                return_tuple = False
            else:
                splits_to_load = [split]
                return_tuple = False
        else:
            splits_to_load = split
            return_tuple = True

        # Validate splits
        valid_splits = {'train', 'val', 'test'}
        invalid_splits = set(splits_to_load) - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid splits: {invalid_splits}. Valid options: {valid_splits}")

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

        # Extract CST features
        logger.info("Extracting CST features...")
        ds = self._extract_cst_features(ds)

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
                    logger.info(f"Using dynamic split sizing: val/test will be limited to {max_split_ratio:.1%} of train size ({train_size} samples)")

            # Second pass: process each requested split
            for split_name in splits_to_load:
                split_ds = ds.filter(lambda x: x['split'] == split_name, num_proc=8)
                original_size = len(split_ds)
                
                if split_name == 'train' and train_subset < 1.0:
                    split_ds = self._get_train_subset(split_ds, train_subset)
                    logger.info(f"Limited train from {original_size} to {len(split_ds)} samples ({train_subset:.1%})")
                
                elif split_name == 'val' and dynamic_split_sizing and train_size is not None:
                    # Calculate target size using ratio if provided, otherwise use legacy logic
                    if val_ratio is not None:
                        max_val_size = int(original_size * val_ratio)
                        logger.info(f"Target val size: {max_val_size} ({val_ratio:.1%} of available val data)")
                    else:
                        max_val_size = int(train_size * max_split_ratio)
                    
                    split_ds = self._limit_split_size(split_ds, max_val_size)
                    if len(split_ds) < original_size:
                        logger.info(f"Limited val from {original_size} to {len(split_ds)} samples")
                
                elif split_name == 'test' and dynamic_split_sizing and train_size is not None:
                    # Calculate target size using ratio if provided, otherwise use legacy logic
                    if test_ratio is not None:
                        max_test_size = int(original_size * test_ratio)
                        logger.info(f"Target test size: {max_test_size} ({test_ratio:.1%} of available test data)")
                    else:
                        max_test_size = int(train_size * max_split_ratio)
                    
                    split_ds = self._limit_split_size(split_ds, max_test_size)
                    if len(split_ds) < original_size:
                        logger.info(f"Limited test from {original_size} to {len(split_ds)} samples")

                # Filter columns
                split_ds = self._filter_columns(split_ds, columns)
                split_datasets.append(split_ds)

            return tuple(split_datasets)
        
        else:
            # Single dataset return
            # Import concatenate_datasets for use in this scope
            from datasets import concatenate_datasets
            
            # Apply train subset if requested and we're loading all data or just train
            if 'train' in splits_to_load and train_subset < 1.0 and dynamic_split_sizing:
                train_split = ds.filter(lambda x: x['split'] == 'train', num_proc=8)
                original_train_size = len(train_split)
                train_split = self._get_train_subset(train_split, train_subset)
                train_size = len(train_split)
                logger.info(f"Limited train from {original_train_size} to {train_size} samples ({train_subset:.1%})")
                
                # Rebuild dataset with limited train split
                from datasets import concatenate_datasets
                non_train = ds.filter(lambda x: x['split'] != 'train', num_proc=8)
                ds = concatenate_datasets([non_train, train_split])
                
                # Apply dynamic sizing to val/test splits
                if val_ratio is None and test_ratio is None:
                    logger.info(f"Using dynamic split sizing: val/test will be limited to {max_split_ratio:.1%} of train size ({train_size} samples)")
                
                # Limit val split
                val_split = ds.filter(lambda x: x['split'] == 'val', num_proc=8)
                original_val_size = len(val_split)
                if val_ratio is not None:
                    max_val_size = int(original_val_size * val_ratio)
                    logger.info(f"Target val size: {max_val_size} ({val_ratio:.1%} of available val data)")
                else:
                    max_val_size = int(train_size * max_split_ratio)
                
                val_split = self._limit_split_size(val_split, max_val_size)
                if len(val_split) < original_val_size:
                    logger.info(f"Limited val from {original_val_size} to {len(val_split)} samples")
                ds = concatenate_datasets([ds.filter(lambda x: x['split'] != 'val', num_proc=8), val_split])
                
                # Limit test split
                test_split = ds.filter(lambda x: x['split'] == 'test', num_proc=8)
                original_test_size = len(test_split)
                if test_ratio is not None:
                    max_test_size = int(original_test_size * test_ratio)
                    logger.info(f"Target test size: {max_test_size} ({test_ratio:.1%} of available test data)")
                else:
                    max_test_size = int(train_size * max_split_ratio)
                
                test_split = self._limit_split_size(test_split, max_test_size)
                if len(test_split) < original_test_size:
                    logger.info(f"Limited test from {original_test_size} to {len(test_split)} samples")
                ds = concatenate_datasets([ds.filter(lambda x: x['split'] != 'test', num_proc=8), test_split])

            # Filter columns and return
            ds = self._filter_columns(ds, columns)
            return ds


if __name__ == "__main__":
    # Example usage
    dataset_loader = CoDeTM4_WithCSTFeatures(cache_dir="data/")
    
    # Test column names access
    logger.info(f"Available columns: {CoDeTM4_WithCSTFeatures.get_column_names()}")
    
    # Load single split with CST features
    logger.info("Loading 10% of training data with CST features...")
    train_subset = dataset_loader.get_dataset(split='train', train_subset=0.1)
    logger.info(f"Train subset size: {len(train_subset)}")
    logger.info(f"Train subset columns: {train_subset.column_names}")
    
    # Load multiple splits with CST features
    logger.info("Loading train, val, and test splits with CST features...")
    train, val, test = dataset_loader.get_dataset(
        split=['train', 'val', 'test'], 
        train_subset=1.0,  # 5% of train data
        dynamic_split_sizing=False,
    )
    logger.info(f"Train dataset size: {len(train)}")
    logger.info(f"Val dataset size: {len(val)}")
    logger.info(f"Test dataset size: {len(test)}")
    logger.info(f"Train dataset columns: {train.column_names}")
    
    # Show sample with CST features
    logger.info("Sample from train dataset with CST features:")
    sample = train[0]
    logger.info(f"Code snippet: {sample['code'][:100]}...")
    if 'cst_features' in sample:
        logger.info(f"CST features (tensor format): {sample['cst_features']}")
    else:
        cst_feature_names = ['function_defs', 'if_statements', 'loops', 'imports', 'comments', 
                           'class_defs', 'max_nesting_depth', 'binary_ops', 'errors']
        cst_values = {name: sample.get(name, 'N/A') for name in cst_feature_names}
        logger.info(f"CST features (individual): {cst_values}")
    
    # Load with specific columns including CST features
    logger.info("Loading specific columns...")
    subset_with_features = dataset_loader.get_dataset(
        split='train', 
        columns=['code', 'target_binary', 'cst_features'] if dataset_loader.features_as_tensor 
                else ['code', 'target_binary', 'function_defs', 'max_nesting_depth'],
        train_subset=0.01
    )
    logger.info(f"Subset with selected columns: {subset_with_features.column_names}")
