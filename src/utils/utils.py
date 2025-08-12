from typing import Dict, Any
from transformers import PreTrainedTokenizer


def tokenize_fn(
    tokenizer: PreTrainedTokenizer, examples: Dict[str, Any], max_length: int = 512
) -> Dict[str, Any]:
    """
    Tokenizes code examples using the provided tokenizer while preserving all columns.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to process the code.
        examples (Dict[str, Any]): Dictionary containing 'code' and other columns.
        max_length (int): Maximum sequence length for tokenization. Defaults to 512.

    Returns:
        Dict[str, Any]: Tokenized inputs with all original columns preserved, target/target_binary at end.

    Raises:
        ValueError: If 'code' key is missing in examples.
    """
    if "code" not in examples:
        raise ValueError("Examples must contain 'code' key")

    tokenized = tokenizer(
        examples["code"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )

    # Create result without target/target_binary first
    examples_without_target = {k: v for k, v in examples.items() if k not in ["target", "target_binary"]}
    result = {**examples_without_target, **tokenized}

    # Add target or target_binary at the end if it exists
    if "target" in examples:
        result["target"] = examples["target"]
    elif "target_binary" in examples:
        result["target_binary"] = examples["target_binary"]

    return result
