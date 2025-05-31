from datasets import load_dataset
from torch.utils.data import random_split
import torch
from dataset import AbstractDataset

class AIGCodeSet(AbstractDataset):
    """
    AIGCodeSet dataset class for loading and processing the AIGCodeSet dataset.
    Inherits from the Dataset class.
    """
    def __init__(self, cache_dir='data/'):
        """
        Initializes the AIGCodeSet dataset class.
        
        Args:
            cache_dir: Directory to cache the dataset.
        """
        super().__init__()
        self.cache_dir = cache_dir

    def _preprocess_dataset(self, dataset):
        """
        Preprocesses the AIGCodeSet dataset by renaming columns and mapping target values.
        
        Args:
            dataset: The dataset to preprocess.
        
        Returns:
            The preprocessed dataset.
        """
        dataset = dataset.select_columns(['code', 'LLM'])
        dataset = dataset.rename_column('LLM', 'target')
        dataset = dataset.map(lambda x: {'target': 'ai' if x['target'] != 'Human' else 'human'}, remove_columns=['target'])
        label_map = {'human': 0, 'ai': 1}
        dataset = dataset.map(lambda x: {'target': label_map[x['target']]})
        return dataset
    
    def _split_dataset(self, dataset):
        """
        Splits the dataset into train, validation, and test sets.
        
        Args:
            dataset: The dataset to split.
        
        Returns:
            A tuple containing the train, validation, and test datasets.
        """
        # Split the dataset into train, validation, and test sets
        train_ds = dataset.train_test_split(test_size=0.2, seed=42)
        train_val = train_ds['train'].train_test_split(test_size=0.1, seed=42)
        
        train = train_val['train']
        val = train_val['test'] 
        test = train_ds['test']
        
        return train, val, test

    def get_dataset(self, split: bool=True):
        """
        Loads the AIGCodeSet dataset, processes it, and returns the train, validation, and test splits.
        """
        # Load the dataset from the Hugging Face hub
        ds = load_dataset("basakdemirok/AIGCodeSet", cache_dir=self.cache_dir, split='train')
        
        # Preprocess the dataset
        ds = self._preprocess_dataset(ds)

        if split:
            # If split is True, split the dataset into train, validation, and test sets
            return self._split_dataset(ds)
        return ds



if __name__ == "__main__":
    # Example usage of the AIGCodeSet class, both with and without splitting
    dataset = AIGCodeSet(cache_dir='data/')
    train, val, test = dataset.get_dataset(split=True)
    print(f"Train dataset size: {len(train)}")
    print(f"Validation dataset size: {len(val)}")
    print(f"Test dataset size: {len(test)}")
    print(f'Sample from train dataset: {train[0]}')
    full_dataset = dataset.get_dataset(split=False)
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Sample from full dataset: {full_dataset[0]}")
