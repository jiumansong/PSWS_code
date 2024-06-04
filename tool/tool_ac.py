import numpy as np
import time
import torch


class Accumulator:  #@save
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:  #@save
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def accuracy(_mode,output, target):  #@save
    if _mode:  
        output = (output > 0.5).float()
        correct = (output == target).float()
        accuracy_bio = correct.sum()
        return accuracy_bio
    else:  
        output = torch.argmax(output, dim=1)   
        correct = (output == target).float()
        accuracy_multi = correct.sum()
        return accuracy_multi

