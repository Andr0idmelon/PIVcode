import torch
import dask.array as da
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule
import joblib
import numpy as np


class MyTrainDataset(Dataset):
    def __init__(self, data_file, label_file,use_len):
        data=joblib.load(data_file)
        self.data = torch.from_numpy((data/255)[0:int(0.8*len(data))]).half()
        self.label = np.load(label_file)[0:int(0.8*len(data))]

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, index):
        data_i = torch.as_tensor(self.data[index])
        data_i1 = torch.as_tensor(self.data[index + 1])
        combined_data = torch.cat([data_i, data_i1], dim=0)
        label = torch.as_tensor(self.label[index]).half()

        return combined_data, label

class MyValDataset(Dataset):
    def __init__(self, data_file, label_file,use_len):
        data=joblib.load(data_file)
        self.data = torch.from_numpy((data/255)[-int(0.2*len(data)):]).half()
        self.label = np.load(label_file)[-int(0.2*len(data)):]

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, index):
        data_i = torch.as_tensor(self.data[index])
        data_i1 = torch.as_tensor(self.data[index + 1])
        combined_data = torch.cat([data_i, data_i1], dim=0)
        
        label = torch.as_tensor(self.label[index]).half()

        return combined_data, label


class MyDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.batchsize = params.data.batchsize
        self.ld_ratios_train = [2]
        self.ld_ratios_val = [2]

    def prepare_data(self):
        pass    

    def setup(self, stage=None):
        self.train_datasets = []
        for ld in self.ld_ratios_train:
            data_file = f'LD{ld}-S10-data.pkl'
            label_file = f'LD{ld}-S10-label-float16.npy'
            dataset = MyTrainDataset(data_file, label_file)
            self.train_datasets.append(dataset)

        self.train_dataset = ConcatDataset(self.train_datasets)

        self.val_datasets = []
        for ld in self.ld_ratios_val:
            data_file = f'LD{ld}-S10-data.pkl'
            label_file = f'LD{ld}-S10-label-float16.npy'
            dataset = MyValDataset(data_file, label_file)
            self.val_datasets.append(dataset)

        self.val_dataset = ConcatDataset(self.val_datasets)


    def train_dataloader(self):
            return DataLoader(dataset=self.train_dataset, batch_size=self.batchsize, num_workers=16, pin_memory=True,persistent_workers=True)

    def val_dataloader(self):
            return DataLoader(dataset=self.val_dataset, batch_size=self.batchsize, num_workers=16, pin_memory=True,persistent_workers=True)


