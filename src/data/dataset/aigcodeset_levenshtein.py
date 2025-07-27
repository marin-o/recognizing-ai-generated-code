import logging
import os
import torch
from typing import Tuple, Union, Dict, List
from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from Levenshtein import distance as Levenshtein_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/gemma-2-2b-it"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise
prompt = "Refine this Python code for me please:\n```python\n{}\n```"

def rewrite_and_calculate_distance(examples: Dict[str, List]) -> Dict[str, List]:
    codes = examples['code']
    formatted_prompts = [prompt.format(code) for code in codes]
    try:
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        rewritten_codes = [
            tokenizer.decode(output, skip_special_tokens=True).replace(formatted_prompts[i], "").strip()
            for i, output in enumerate(outputs)
        ]
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        rewritten_codes = codes  # Fallback to original
    distances = [Levenshtein_distance(rewritten, original) for rewritten, original in zip(rewritten_codes, codes)]
    return {**examples, 'rewritten': rewritten_codes, 'distance': distances}

class AIGCodeSet_Levenshtein:
    def __init__(self, cache_dir: str = 'data/'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _calculate_distances(self, data: Dataset) -> Dataset:
        return data.map(rewrite_and_calculate_distance, batched=True, batch_size=4)  # Reduced for 8GB VRAM

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        if not all(col in dataset.column_names for col in ["code", "LLM"]):
            raise ValueError("Dataset missing required columns: 'code' and/or 'LLM'")
        dataset = dataset.select_columns(["code", "LLM"])
        dataset = dataset.rename_column("LLM", "target")
        dataset = dataset.map(lambda x: {"target": "ai" if x["target"] != "Human" else "human"})
        label_map = {"human": 0, "ai": 1}
        dataset = dataset.map(lambda x: {"target": label_map[x["target"]]})
        dataset = dataset.cast_column("target", ClassLabel(names=["human", "ai"]))
        dataset = self._calculate_distances(dataset)
        feature_columns = [col for col in dataset.column_names if col not in ["code", "target"]]
        final_columns = ["code"] + feature_columns + ["target"]
        dataset = dataset.select_columns(final_columns)
        return dataset

    def _split_dataset(self, dataset: Dataset, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        if not (0 < test_size < 1 and 0 < val_size < 1):
            raise ValueError("test_size and val_size must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")
        train_ds = dataset.train_test_split(test_size=test_size, seed=42, stratify_by_column="target")
        train_val = train_ds["train"].train_test_split(
            test_size=val_size / (1 - test_size), seed=42, stratify_by_column="target"
        )
        return train_val["train"], train_val["test"], train_ds["test"]

    def get_dataset(self, split: bool = True, test_size: float = 0.2, val_size: float = 0.1) -> Union[Dataset, Tuple[Dataset, Dataset, Dataset]]:
        try:
            ds = load_dataset("basakdemirok/AIGCodeSet", split="train")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
        ds = self._preprocess_dataset(ds)
        if split:
            return self._split_dataset(ds, test_size, val_size)
        return ds

if __name__ == '__main__':
    dataset = AIGCodeSet_Levenshtein()
    train, val, test = dataset.get_dataset(split=True)
    logger.info(f"Train dataset size: {len(train)}")
    logger.info(f"Validation dataset size: {len(val)}")
    logger.info(f"Test dataset size: {len(test)}")
    print(f"Train dataset: {train}")
    logger.info(f"Sample from train dataset: {train[1000]}")
    full_dataset = dataset.get_dataset(split=False)
    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Sample from full dataset: {full_dataset[0]}")
