import logging
import os
import torch
import json
from typing import Tuple, Union, Dict, List
from datasets import load_dataset, Dataset, ClassLabel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AIGCodeSet_Levenshtein:
    def __init__(self, cache_dir: str = 'data/', cache_file: str = None):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Set default cache file path
        self.cache_file = cache_file or os.path.join(cache_dir, 'aigcodeset_perturbations_levenshtein.json')
        
        # Move all heavy initialization here
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/gemma-2-2b-it"
        self.prompt = "Refine this Python code for me please:\n```python\n{}\n```"
        
        # Initialize model and tokenizer lazily
        self.model = None
        self.tokenizer = None

    def _load_cached_dataset(self) -> Dataset:
        """Load dataset from cached JSON file if it exists"""
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached dataset from {self.cache_file}")
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to Dataset format
                dataset = Dataset.from_list(list(data.values()))
                
                # Fix: Re-cast target column to ClassLabel after loading from JSON
                if 'target' in dataset.column_names:
                    dataset = dataset.cast_column("target", ClassLabel(names=["human", "ai"]))
                
                logger.info(f"Successfully loaded cached dataset with {len(dataset)} samples")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {str(e)}. Will regenerate.")
                return None
        return None

    def _save_dataset_cache(self, dataset: Dataset):
        """Save dataset to JSON cache file"""
        try:
            # Convert dataset to dictionary format for JSON serialization
            data_dict = {}
            for i, sample in enumerate(dataset):
                data_dict[str(i)] = sample
            
            with open(self.cache_file, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            logger.info(f"Dataset cached to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {str(e)}")
        
    def _load_model(self):
        """Load model and tokenizer only when needed"""
        if self.model is not None:
            return  # Already loaded
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _rewrite_and_calculate_distance(self, examples: Dict[str, List]) -> Dict[str, List]:
        # Load model only when actually needed
        self._load_model()
        
        codes = examples['code']
        formatted_prompts = [self.prompt.format(code) for code in codes]
        
        try:
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1
            )
            
            rewritten_codes = [
                self.tokenizer.decode(output, skip_special_tokens=True).replace(formatted_prompts[i], "").strip()
                for i, output in enumerate(outputs)
            ]
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            rewritten_codes = codes  # Fallback to original
        
        from Levenshtein import distance as Levenshtein_distance
        distances = [Levenshtein_distance(rewritten, original) for rewritten, original in zip(rewritten_codes, codes)]
        return {**examples, 'rewritten': rewritten_codes, 'levenshtein_distance': distances}

    def _calculate_distances(self, data: Dataset) -> Dataset:
        return data.map(self._rewrite_and_calculate_distance, batched=True, batch_size=4)

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        if not all(col in dataset.column_names for col in ["code", "LLM"]):
            raise ValueError("Dataset missing required columns: 'code' and/or 'LLM'")
        
        # Check for cached version first
        cached_dataset = self._load_cached_dataset()
        if cached_dataset is not None:
            return cached_dataset
        
        # If no cache, process normally
        logger.info("No cached dataset found. Processing dataset from scratch...")
        dataset = dataset.select_columns(["code", "LLM"])
        dataset = dataset.rename_column("LLM", "target")
        dataset = dataset.map(lambda x: {"target": "ai" if x["target"] != "Human" else "human"})
        label_map = {"human": 0, "ai": 1}
        dataset = dataset.map(lambda x: {"target": label_map[x["target"]]})
        dataset = dataset.cast_column("target", ClassLabel(names=["human", "ai"]))
        
        # Add original_code column before processing
        dataset = dataset.map(lambda x: {"original_code": x["code"]})
        
        # Calculate distances (this is the expensive part)
        dataset = self._calculate_distances(dataset)
        
        # Reorder columns
        feature_columns = [col for col in dataset.column_names if col not in ["code", "target"]]
        final_columns = ["original_code"] + feature_columns + ["target"]
        dataset = dataset.select_columns(final_columns)
        
        # Save to cache for future use
        self._save_dataset_cache(dataset)
        
        return dataset

    def _split_dataset(self, dataset: Dataset, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        if not (0 < test_size < 1 and 0 < val_size < 1):
            raise ValueError("test_size and val_size must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")
        
        # Ensure target column is ClassLabel before stratification
        if not isinstance(dataset.features['target'], ClassLabel):
            dataset = dataset.cast_column("target", ClassLabel(names=["human", "ai"]))
        
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