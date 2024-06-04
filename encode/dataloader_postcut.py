import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


def label_int(_label, _class_num, _class_cate):
    for i in range(_class_num):  
        if str(_label) == list(_class_cate)[i]:
            conv_label = i
            return conv_label


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, class_cate=None, class_num=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self._class_cate = class_cate
        self._class_number = class_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]

        slice_path = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        label = self.data.iloc[idx, 1]
        _label = label_int(label, self._class_number, self._class_cate)
        img = Image.open(img_path)
        patches = []
        for i in range(9):
            for j in range(9):
                patch = img.crop((i * 224, j * 224, (i + 1) * 224, (j + 1) * 224))
                if self.transform:
                    patch = self.transform(patch)
                patches.append(patch)
        patches = torch.stack(patches)  

        return {"patches": patches, "_label": _label, "slice_path": slice_path, "img_name": img_name}



def load_data(_class_cate, _class_number, batch_size):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
    ])
    test_csv_path = 'CSV path that stores the pair of train patch paths and labels'
    test_dataset = CustomDataset(csv_file=test_csv_path, transform=data_transforms, class_cate=_class_cate, class_num=_class_number)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    train_csv_path = 'CSV path that stores the pair of test patch paths and labels'
    train_dataset = CustomDataset(csv_file=train_csv_path, transform=data_transforms, class_cate=_class_cate, class_num=_class_number)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


