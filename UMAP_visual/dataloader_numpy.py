#dataloader先对一个大的patch中的所有2048的向量取均值。得到1x1048。然后遍历所有的npy文件，最后绘制成一个numpy数组，每一行代表一个样本。
import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class FeaturesDataset_test_numpy(Dataset):
    def __init__(self, path, mode='test'):
        super().__init__()
        self.mode = mode
        self.data = []
        self.label = []
        # 创建 PCA 模型，将高维向量降至 32 维
        if self.mode == 'test':
            folders = sorted(glob.glob(os.path.join(path, mode, '*', '*')))
            n_folders = len(folders)
            for i in range(n_folders):
                folder_path = folders[i]
                npy_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg.npy')))
                for fname in npy_files:
                    npy = np.load(fname)
                    #print(npy.shape)
                    npy = np.mean(npy, axis=0)
                    npy = npy.reshape(2048)
                    #print(npy.shape)
                # 将列表转换为 Numpy 数组
                    folder_data = np.array(npy)
                # 计算大的patch的小patch样本特征向量的均值
                    self.data.append(folder_data)
                    parts = folder_path.split(os.path.sep)
                    label = int(parts[-2])
                    self.label.append(label)
            self.label = np.vstack(self.label)
            self.data = np.vstack(self.data)

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data, self.label
    #直接返回全部数据

def load_data():
    path = '/fs1/private/user/songjiuman/C2C/patch2016_2x_sumclass_macenko_imagnet'
    test_dataset_class = FeaturesDataset_test_numpy(path, mode='test')
    test_dataset, test_labels = test_dataset_class.get_data()
    print(test_dataset.shape)
    print(test_labels.shape)
    return test_dataset, test_labels

load_data()
