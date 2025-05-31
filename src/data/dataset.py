from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    
    @abstractmethod
    def _preprocess_dataset():
        """
        Abstract method to preprocess the dataset.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _split_dataset():
        """
        Abstract method to split the dataset.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_dataset(self, split: bool=False):
        """
        Abstract method to get the dataset.
        Must be implemented by subclasses.
        """
        pass

