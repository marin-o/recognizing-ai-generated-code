import logging
import os
from typing import Tuple, Union, List, Dict
from datasets import Dataset, load_from_disk, ClassLabel
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Language-specific node types for nesting depth calculation
NESTING_NODES = {
    'python': {
        "function_definition", "class_definition", "if_statement", 
        "for_statement", "while_statement", "with_statement", "try_statement"
    },
    'cpp': {
        "function_definition", "class_specifier", "struct_specifier", 
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "namespace_definition", "try_statement"
    },
    'java': {
        "method_declaration", "constructor_declaration", "class_declaration", 
        "interface_declaration", "if_statement", "for_statement", "while_statement", 
        "enhanced_for_statement", "switch_statement", "try_statement"
    }
}

# Language-specific queries
LANGUAGE_QUERIES = {
    'python': {
        'functions': "(function_definition) @func",
        'classes': "(class_definition) @class_def",
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt"],
        'imports': ["(import_statement) @import", "(import_from_statement) @import_from"],
        'comments': "(comment) @comment",
        'binary_ops': "(binary_operator) @binop",
        'errors': "(ERROR) @error"
    },
    'cpp': {
        'functions': "(function_definition) @func",
        'classes': ["(class_specifier) @class_def", "(struct_specifier) @struct_def"],
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt"],
        'imports': ["(preproc_include) @include", "(using_declaration) @using"],
        'comments': ["(comment) @comment", "(preproc_comment) @preproc_comment"],
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'java': {
        'functions': ["(method_declaration) @method", "(constructor_declaration) @constructor"],
        'classes': ["(class_declaration) @class_def", "(interface_declaration) @interface_def"],
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt", "(enhanced_for_statement) @enhanced_for"],
        'imports': "(import_declaration) @import",
        'comments': ["(line_comment) @line_comment", "(block_comment) @block_comment"],
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    }
}


def get_language_parser(language: str) -> Tuple[Language, Parser]:
    """Get the appropriate Tree-sitter language and parser for a given language."""
    if language.lower() == 'python':
        lang = Language(tspython.language())
    elif language.lower() in ['cpp', 'c++']:
        lang = Language(tscpp.language())
    elif language.lower() == 'java':
        lang = Language(tsjava.language())
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    parser = Parser(lang)
    return lang, parser


def traverse(node: Node, depth: int, language: str) -> int:
    """
    Traverse the CST and compute the maximum nesting depth.

    Args:
        node (Node): The current node in the CST.
        depth (int): The current depth in the tree.
        language (str): The programming language.

    Returns:
        int: The maximum nesting depth encountered.
    """
    max_nesting_depth = depth
    
    node_type = node.type
    nesting_types = NESTING_NODES.get(language.lower(), set())
    
    if node_type in nesting_types:
        for child in node.children:
            max_nesting_depth = max(max_nesting_depth, traverse(child, depth + 1, language))
    else:
        for child in node.children:
            max_nesting_depth = max(max_nesting_depth, traverse(child, depth, language))

    return max_nesting_depth


def execute_queries(lang: Language, root_node: Node, queries: List[str]) -> int:
    """Execute multiple Tree-sitter queries and return total count."""
    total_count = 0
    for query_str in queries:
        try:
            query = lang.query(query_str)
            total_count += len(query.captures(root_node))
        except Exception as e:
            logger.debug(f"Query failed: {query_str}, error: {e}")
            # Continue with other queries instead of failing completely
            continue
    return total_count


def safe_execute_single_query(lang: Language, root_node: Node, query_str: str) -> int:
    """Execute a single Tree-sitter query safely."""
    try:
        query = lang.query(query_str)
        return len(query.captures(root_node))
    except Exception as e:
        logger.debug(f"Query failed: {query_str}, error: {e}")
        return 0


def extract_features_for_example(example: Dict, as_tensor: bool, flag=False) -> Dict:
    """
    Extract CST features for a single example, supporting multiple languages.

    Args:
        example (Dict): A single dataset example containing 'code' and 'language'.
        as_tensor (bool): Whether to return features as a tensor-like list.

    Returns:
        Dict: The example with added CST features.
    """
    # Default features for empty or failed cases
    default_features = {
        'function_defs': 0, 'class_defs': 0, 'if_statements': 0,
        'loops': 0, 'imports': 0, 'comments': 0, 'binary_ops': 0,
        'errors': 1, 'max_nesting_depth': 0,  # errors=1 since unparseable code indicates syntax errors
    }
    
    try:
        # Get language and code
        language = example.get('language', 'python').lower()
        code = example.get('cleaned_code', '')
        
        if not code:
            logger.warning("Empty code found, using default features")
            # Still add default features to maintain consistent schema
            if as_tensor:
                example["cst_features"] = list(default_features.values())
            else:
                example.update(default_features)
            return example
        
        # Get appropriate parser
        lang, parser = get_language_parser(language)
        
        # Parse code
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        
        # Get language-specific queries
        queries = LANGUAGE_QUERIES.get(language, LANGUAGE_QUERIES['python'])
        
        # Calculate nesting depth
        max_nesting_depth = traverse(root_node, 0, language)
        
        # Extract features using queries
        features = {}
        
        # Functions
        if isinstance(queries['functions'], list):
            features['function_defs'] = execute_queries(lang, root_node, queries['functions'])
        else:
            features['function_defs'] = safe_execute_single_query(lang, root_node, queries['functions'])
        
        # Classes
        if isinstance(queries['classes'], list):
            features['class_defs'] = execute_queries(lang, root_node, queries['classes'])
        else:
            features['class_defs'] = safe_execute_single_query(lang, root_node, queries['classes'])
        
        # If statements
        features['if_statements'] = safe_execute_single_query(lang, root_node, queries['if_statements'])
        
        # Loops
        features['loops'] = execute_queries(lang, root_node, queries['loops'])
        
        # Imports
        if isinstance(queries['imports'], list):
            features['imports'] = execute_queries(lang, root_node, queries['imports'])
        else:
            features['imports'] = safe_execute_single_query(lang, root_node, queries['imports'])
        
        # Comments
        if isinstance(queries['comments'], list):
            features['comments'] = execute_queries(lang, root_node, queries['comments'])
        else:
            features['comments'] = safe_execute_single_query(lang, root_node, queries['comments'])
        
        # Binary operations
        features['binary_ops'] = safe_execute_single_query(lang, root_node, queries['binary_ops'])
        
        # Errors
        features['errors'] = safe_execute_single_query(lang, root_node, queries['errors'])
        
        # Nesting depth
        features['max_nesting_depth'] = max_nesting_depth
    
        
        if as_tensor:
            # Convert to tensor format
            features = {"cst_features": list(features.values())}
        
        example.update(features)
        return example
        
    except Exception as e:
        logger.warning(f"Failed to extract CST features for {example.get('language', 'unknown')} code: {str(e)}")
        # Return example with default features on failure to maintain consistent schema
        if as_tensor:
            example["cst_features"] = list(default_features.values())
        else:
            example.update(default_features)
        return example


class CoDeTM4CleanedWithCSTFeatures:
    """Dataset class for loading cleaned CoDet-M4 dataset with multi-language CST features."""

    # Dataset column names - available even before loading
    COLUMN_NAMES = [
        'code', 'target', 'model', 'language', 'source', 'features', 'cleaned_code', 'split'
    ]

    def __init__(self, cleaned_data_path: str, features_as_tensor: bool = True):
        """
        Initialize the CoDeTM4CleanedWithCSTFeatures dataset class.

        Args:
            cleaned_data_path (str): Path to the directory containing cleaned datasets.
            features_as_tensor (bool): Whether to return CST features as tensor-like list.

        Raises:
            ValueError: If cleaned_data_path is invalid or does not exist.
            FileNotFoundError: If required split directories are not found.
        """
        if not os.path.isdir(cleaned_data_path):
            raise ValueError(
                f"Cleaned data path '{cleaned_data_path}' does not exist or is not a directory"
            )
        
        self.cleaned_data_path = cleaned_data_path
        self.features_as_tensor = features_as_tensor
        
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
        
        logger.info(f"Initialized CoDeTM4CleanedWithCSTFeatures with data from: {cleaned_data_path}")

    @staticmethod
    def get_column_names() -> List[str]:
        """Get the list of available column names in the CoDet-M4 dataset."""
        return CoDeTM4CleanedWithCSTFeatures.COLUMN_NAMES.copy()

    def _load_split(self, split_name: str) -> Dataset:
        """Load a specific split from disk."""
        if split_name not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split_name}'. Must be one of ['train', 'val', 'test']")
        
        try:
            split_path = os.path.join(self.cleaned_data_path, split_name)
            dataset = load_from_disk(split_path)
            
            # Ensure target_binary column exists with proper type
            if 'target_binary' not in dataset.column_names:
                logger.info(f"Adding target_binary column to {split_name} split...")
                def map_target_binary(example, flag=True):
                    target_binary = 0 if example["target"] == "human" else 1
                    return {"target_binary": target_binary}
                os.makedirs('data/codetcst/')
                dataset = dataset.map(map_target_binary, num_proc=10, cache_file_name="data/codetcst/cache01.arrow")
                dataset = dataset.cast_column("target_binary", ClassLabel(names=["human", "ai"]))
            
            logger.info(f"Loaded {split_name} split: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {split_name} split from {split_path}: {str(e)}")

    def _extract_cst_features(self, data: Dataset, split_name: str = "unknown") -> Dataset:
        """Extract CST features for all examples in the dataset."""
        logger.info("Extracting multi-language CST features...")
        
        # Check languages in dataset
        if 'language' in data.column_names:
            languages = set(data['language'])
            logger.info(f"Found languages in dataset: {languages}")
        os.makedirs('data/codetcst/', exist_ok=True)
        return data.map(
            lambda x: extract_features_for_example(x, as_tensor=self.features_as_tensor),
            num_proc=10,
            desc="Extracting CST features",
            cache_file_name=f'data/codetcst/cache_{split_name}.arrow'
        )

    def _filter_columns(self, dataset: Dataset, columns: Union[str, List[str]]) -> Dataset:
        """Filter dataset columns based on the columns parameter."""
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
        """Get a stratified subset of the training dataset."""
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
            logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
            total_samples = len(dataset)
            subset_size = int(total_samples * subset_fraction)
            
            # Use random sampling instead of sequential sampling to maintain class balance
            import numpy as np
            np.random.seed(42)
            indices = np.random.choice(total_samples, size=subset_size, replace=False)
            indices = sorted(indices)  # Sort for deterministic behavior
            return dataset.select(indices)

    def _limit_split_size(self, dataset: Dataset, max_size: int) -> Dataset:
        """Limit the size of a dataset split using stratified sampling."""
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
                # Use random sampling instead of sequential sampling
                import numpy as np
                np.random.seed(42)
                indices = np.random.choice(len(dataset), size=max_size, replace=False)
                indices = sorted(indices)  # Sort for deterministic behavior
                return dataset.select(indices)
        except Exception as e:
            logger.warning(f"Stratified sampling failed for split limiting: {e}. Using random sampling.")
            # Use random sampling instead of sequential sampling
            import numpy as np
            np.random.seed(42)
            indices = np.random.choice(len(dataset), size=max_size, replace=False)
            indices = sorted(indices)  # Sort for deterministic behavior
            return dataset.select(indices)
            return dataset.select(range(max_size))

    def get_dataset(
        self, 
        split: Union[str, List[str]] = 'all', 
        columns: Union[str, List[str]] = 'all', 
        train_subset: float = 1.0,
        dynamic_split_sizing: bool = False,
        max_split_ratio: float = 0.2,
        val_ratio: float = None,
        test_ratio: float = None
    ) -> Union[Dataset, Tuple[Dataset, ...]]:
        """
        Load and process the cleaned CoDet-M4 dataset with multi-language CST features.

        Args:
            split (Union[str, List[str]]): Dataset split(s) to load.
            columns (Union[str, List[str]]): Columns to include.
            train_subset (float): Fraction of training data to load (0.0 to 1.0).
            dynamic_split_sizing (bool): Whether to dynamically limit val/test sizes.
            max_split_ratio (float): Maximum ratio of val/test size to train size.
            val_ratio (float, optional): Specific ratio for validation set size.
            test_ratio (float, optional): Specific ratio for test set size.

        Returns:
            Union[Dataset, Tuple[Dataset, ...]]: Dataset(s) with CST features.
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
            # Load each split independently
            dataset = self._load_split(split_name)
            
            # Extract CST features
            dataset = self._extract_cst_features(dataset, split_name)
            
            datasets[split_name] = dataset

        # Apply train subset if needed
        if 'train' in datasets and train_subset < 1.0:
            datasets['train'] = self._get_train_subset(datasets['train'], train_subset)
            
        # Update cache after all processing is complete
        for split_name in datasets:
            if split_name == 'train':
                self._train = datasets['train']
            elif split_name == 'val':
                self._val = datasets['val']
            elif split_name == 'test':
                self._test = datasets['test']

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
        """Get information about the dataset."""
        info = {
            'cleaned_data_path': self.cleaned_data_path,
            'features_as_tensor': self.features_as_tensor,
            'available_splits': ['train', 'val', 'test'],
            'column_names': self.get_column_names(),
            'supported_languages': ['python', 'cpp', 'java']
        }
        
        # Add size information if datasets are cached
        if self._train is not None:
            info['train_size'] = len(self._train)
        if self._val is not None:
            info['val_size'] = len(self._val)
        if self._test is not None:
            info['test_size'] = len(self._test)
        
        return info


if __name__ == "__main__":
    # Example usage
    CLEANED_DATA_PATH = "data/codet_cleaned_20250812_201438/"  # Update this path
    
    if os.path.exists(CLEANED_DATA_PATH):
        # Initialize dataset loader
        dataset_loader = CoDeTM4CleanedWithCSTFeatures(
            cleaned_data_path=CLEANED_DATA_PATH, 
            features_as_tensor=True
        )
        
        # Get dataset info
        info = dataset_loader.get_info()
        logger.info(f"Dataset info: {info}")
        
        # Load small subset to test multi-language CST features
        logger.info("Loading small subset with CST features...")
        train_small = dataset_loader.get_dataset(split='train', train_subset=0.01)
        logger.info(f"Small train dataset size: {len(train_small)}")
        logger.info(f"Columns: {train_small.column_names}")
        
        # Show sample with CST features for different languages
        for i in range(min(3, len(train_small))):
            sample = train_small[i]
            logger.info(f"Sample {i+1}: Language={sample.get('language', 'unknown')}")
            if 'cst_features' in sample:
                logger.info(f"  CST features (tensor): {sample['cst_features']}")
            else:
                cst_keys = ['function_defs', 'class_defs', 'if_statements', 'loops', 
                           'imports', 'comments', 'binary_ops', 'errors', 
                           'max_nesting_depth']
                cst_values = {k: sample.get(k, 'N/A') for k in cst_keys}
                logger.info(f"  CST features: {cst_values}")
        
    else:
        logger.error(f"Cleaned data path not found: {CLEANED_DATA_PATH}")
        logger.info("Please make sure the cleaned data exists at the specified path.")
