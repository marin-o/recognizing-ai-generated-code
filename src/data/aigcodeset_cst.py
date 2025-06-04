import logging
import os
from typing import Tuple, Union, Dict
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


def extract_features_for_example(example: Dict) -> Dict:
    """
    Extract CST features for a single example.

    Args:
        example (Dict): A single dataset example containing 'code'.

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
    features = {"features": list(features.values())}
    example.update(features)
    return example


class AIGCodeSet_WithCSTFeatures:
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

    def _extract_cst_features(self, data: Dataset) -> Dataset:
        """
        Extract CST features for all examples in the dataset.

        Args:
            data (Dataset): The dataset to process.

        Returns:
            Dataset: The dataset with added CST features.
        """
        return data.map(
            extract_features_for_example,
            num_proc=4,  # Parallelize across 4 processes
        )

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset by selecting columns, renaming, mapping labels, and reordering columns.

        Args:
            dataset (Dataset): The raw dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset with 'target' as the last column.
        """
        if not all(col in dataset.column_names for col in ["code", "LLM"]):
            raise ValueError("Dataset missing required columns: 'code' and/or 'LLM'")
        dataset = dataset.select_columns(["code", "LLM"])
        dataset = dataset.rename_column("LLM", "target")
        dataset = dataset.map(
            lambda x: {"target": "ai" if x["target"] != "Human" else "human"}
        )
        label_map = {"human": 0, "ai": 1}
        dataset = dataset.map(lambda x: {"target": label_map[x["target"]]})
        dataset = dataset.cast_column("target", ClassLabel(names=["human", "ai"]))
        dataset = self._extract_cst_features(dataset)

        # Reorder columns to place 'target' last
        feature_columns = [
            col for col in dataset.column_names if col not in ["code", "target"]
        ]
        final_columns = ["code"] + feature_columns + ["target"]
        dataset = dataset.select_columns(final_columns)
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
    dataset = AIGCodeSet_WithCSTFeatures(cache_dir="data/")
    train, val, test = dataset.get_dataset(split=True)
    logger.info(f"Train dataset size: {len(train)}")
    logger.info(f"Validation dataset size: {len(val)}")
    logger.info(f"Test dataset size: {len(test)}")
    print(f"Train dataset: {train}")
    logger.info(f"Sample from train dataset: {train[1000]}")
    full_dataset = dataset.get_dataset(split=False)
    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Sample from full dataset: {full_dataset[0]}")
