from typing import Dict, Any
from transformers import PreTrainedTokenizer


def tokenize_fn(
    tokenizer: PreTrainedTokenizer, examples: Dict[str, Any], max_length: int = 512
) -> Dict[str, Any]:
    """
    Tokenizes code examples using the provided tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to process the code.
        examples (Dict[str, Any]): Dictionary containing 'code' and 'target' keys.
        max_length (int): Maximum sequence length for tokenization. Defaults to 512.

    Returns:
        Dict[str, Any]: Tokenized inputs with 'input_ids', 'attention_mask', and 'target'.

    Raises:
        ValueError: If 'code' or 'target' keys are missing in examples.
    """
    if "code" not in examples or "target" not in examples:
        raise ValueError("Examples must contain 'code' and 'target' keys")

    tokenized = tokenizer(
        examples["code"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )
    tokenized["target"] = examples["target"]
    return tokenized
