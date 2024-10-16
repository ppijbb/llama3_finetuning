from typing import Iterable
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class DistilDataset(Dataset):
    def __init__(self, data: Iterable, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class DistillationDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32, num_workers=4):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        self.train_dataset = DistilDataset(self.train_data, transform=self.transform)
        self.val_dataset = DistilDataset(self.val_data, transform=self.transform)
        self.test_dataset = DistilDataset(self.test_data, transform=self.transform)

    def _get_data_loader(self, dataset):
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            shuffle=True if dataset==self.train_dataset else False,
            num_workers=self.num_workers if dataset==self.train_dataset else 1)

    def train_dataloader(self):
        return self._get_data_loader(dataset=self.train_dataset)

    def val_dataloader(self):
        return self._get_data_loader(dataset=self.val_dataset)

    def test_dataloader(self):
        return self._get_data_loader(dataset=self.test_dataset)
