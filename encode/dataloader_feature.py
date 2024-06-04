import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self,path,mode='train',level=1):
        super().__init__()
        self.mode=mode
        self.data=[]
        self.label=[]

        if mode == 'train' or self.mode == 'test':
            filenames_train = sorted(glob.glob(path+'train/*'+'/*/*'+'.jpg.npy'))
            filenames_test = sorted(glob.glob(path+'test/*'+'/*/*'+'.jpg.npy'))
            n_train = len(filenames_train)
            n_test = len(filenames_test)

            if mode == 'train':
                for i in range(n_train):
                    fname = filenames_train[i]
                    npy = np.load(fname)
                    self.data.append(npy)
                    parts = fname.split(os.path.sep)  
                    label = parts[-3]
                    label = int(label)
                    self.label.append(label)

            if mode == 'test':
                for i in range(n_test):
                    fname = filenames_test[i]
                    npy = np.load(fname)
                    self.data.append(npy)
                    parts = fname.split(os.path.sep)  
                    label = parts[-3]
                    label = int(label)
                    self.label.append(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def load_data_feature(_class_cate, _class_number, _batch_size):

    path = 'the path of patches and features'
    test_dataset = FeaturesDataset(path, mode='test')
    test_loader = DataLoader(test_dataset,  batch_size=_batch_size, shuffle=True, drop_last=True)

    train_dataset = FeaturesDataset(path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


# Load the feature vectors of all patches of wsi at once
class FeaturesDataset_test_wsi(Dataset):
    def __init__(self, path, mode='test', level=1):
        super().__init__()
        self.mode = mode
        self.data = []
        self.label = []
        self.path = []
        if self.mode == 'test':
            folders = sorted(glob.glob(os.path.join(path, mode, '*', '*')))
            n_folders = len(folders)

            for i in range(n_folders):
                folder_path = folders[i]
                npy_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg.npy')))

                folder_data = []
                for fname in npy_files:
                    npy = np.load(fname)
                    folder_data.append(npy)
                folder_data_tensor = torch.tensor(np.stack(folder_data))  
                self.data.append(folder_data_tensor)
                parts = folder_path.split(os.path.sep)
                label = int(parts[-2])
                self.label.append(label)
                self.path.append(folder_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.path[index]


def load_data_feature_test_wsi():
    path = 'the path of test patches and features'
    test_dataset = FeaturesDataset_test_wsi(path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)   

    return test_loader
