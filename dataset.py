import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


class CoDETM4DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/codet/",
        batch_size: int = 32,
        cleaned_code: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cleaned_code = cleaned_code

    def prepare_data(self):
        load_dataset("DaniilOr/CoDET-M4", cache_dir=self.data_dir)

    def setup(self, stage: str):
        code = "cleaned_code" if self.cleaned_code else "code"
        ds_full = load_dataset(
            "DaniilOr/CoDET-M4", cache_dir=self.data_dir
        ).select_columns([code, "target"])["train"]
        train_set, test_set = random_split(
            ds_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
        train_set, val_set = random_split(
            train_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

        if stage == "fit":
            self.train_set = train_set
            self.val_set = val_set

        if stage == "test":
            self.test_set = ds_full

        if stage == "predict":
            self.predict_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    dm = CoDETM4DataModule(data_dir="data/codet/", batch_size=32, cleaned_code=False)
    dm.prepare_data()
    dm.setup(stage="fit")
    print(f"Train set size: {len(dm.train_set)}, Val set size: {len(dm.val_set)}")
    print(f"Train sample: {dm.train_set[0]}")
    print(f"Val sample: {dm.val_set[0]}")
    dm.setup(stage="test")
    print(f"Test sample: {dm.test_set[0]}")
    dm.setup(stage="predict")
    print(f"Predict sample: {dm.predict_set[0]}")
