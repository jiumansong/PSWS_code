import os
import torch
import random
import numpy as np
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
import time
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tool.tool_ac import Timer, Accumulator, accuracy
from encode.dataloader_feature import load_data_feature
from torch import optim
from tool.model_1024 import ps_vit_1024_16head
import math
from statistics import mode
import gc
import pynvml

parser = argparse.ArgumentParser(description='C2C Classify')
parser.add_argument('--name', default='C2C Classify', type=str)   
parser.add_argument('--EPOCH', default=50, type=int)  
parser.add_argument('--epoch_step', default='[50]', type=str)    
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--lr', default=1e-5, type=float)    
parser.add_argument('--weight_decay', default=1e-3, type=float)    
parser.add_argument('--lr_decay_ratio', default=0.1, type=float)  
parser.add_argument('--log_dir', default='debug_log', type=str)   
parser.add_argument('--class_num', default=4, type=int)   
parser.add_argument('--num_samples', default=16, type=int)   #batch_size

jobid = str(time.strftime('%m%d-%H%M%S', time.localtime(time.time())))


torch.manual_seed(15)
torch.cuda.manual_seed(15)
np.random.seed(15)
random.seed(15)


def print_log(tstr, f):
    f.write('\n')
    f.write(tstr)
    print(tstr)


class ConnectionModule(torch.nn.Module):
    def __init__(self):
        super(ConnectionModule, self).__init__()

    def forward(self, features):
        tensor_2d = torch.cat([features], dim=0)  
        num_vectors = tensor_2d.shape[0]  
        max_square = int(math.sqrt(num_vectors)) ** 2
        height, width = int(math.sqrt(max_square)), int(math.sqrt(max_square))
        feature_map = tensor_2d.view(height, width, -1)  
        FeatureMap = feature_map.permute(2, 0, 1)
        return FeatureMap   


def main():
    params = parser.parse_args()
    epoch_step = json.loads(params.epoch_step)
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))  

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)  
    log_dir = os.path.join(params.log_dir, 'log.txt')  
    save_dir = os.path.join(params.log_dir, 'The path to store the optimal model.pth')  

    z = vars(params).copy()  
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))  
    log_file = open(log_dir, 'a')  

    class_cate = ['0', '1', '2', '3']
    class_number = params.class_num
    train_loader, val_loader = load_data_feature(class_cate, class_number, params.num_samples)

    Classifier = ps_vit_1024_16head().to(params.device)
    print(Classifier)
    Connection = ConnectionModule().to(params.device)

    classifier_parameters = []
    classifier_parameters += list(Classifier.parameters())

    for param in classifier_parameters:  
        param.requires_grad = True

    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean').to(params.device) 
    optimizer_classifier = optim.Adam(classifier_parameters, lr=params.lr, weight_decay=params.weight_decay)
    scheduler_classifier = torch.optim.lr_scheduler.MultiStepLR(optimizer_classifier, epoch_step, gamma=params.lr_decay_ratio)

    avg_training_loss = []
    avg_val_loss = []
    best_auc = 0   
    best_epoch = -1   

    print_log(f'>>>>>>>>>>> Model training start ...', log_file)
    for epoch in range(params.EPOCH):
        print_log(f'>>>>>>>>>>> Training Epoch: {epoch}', log_file)

        training_loss, metric, timer = model_train(classifier=Classifier, connection=Connection,
                                                   num_samples=params.num_samples,
                                                   train_loader=train_loader,
                                                   criterion=ce_loss,
                                                   optimizer=optimizer_classifier,
                                                   params=params, f_log=log_file)

        print_log(f'>>>>>>>>>>> Val Epoch: {epoch}', log_file)  
        val_loss, metric2 = bag_model_validation(classifier=Classifier, connection=Connection,
                                             val_loader=val_loader,
                                             criterion=ce_loss,  params=params, f_log=log_file)

        training_loss_overall = np.average(training_loss)  
        val_loss_overall = np.average(val_loss)  
        scheduler_classifier.step()
        avg_training_loss.append(training_loss_overall)  
        avg_val_loss.append(val_loss_overall)

        epoch_len = len(str(params.EPOCH))
        print_msg = (f'[{epoch:>{epoch_len}}/{params.EPOCH:>{epoch_len}}] ' +
                     f'train_loss: {training_loss_overall:.5f} ' +
                     f'val_loss: {val_loss_overall:.5f}')
        print_log(print_msg, log_file)

        train_acc = metric[0] / metric[1]
        val_acc = metric2[0] / metric2[1]
        print_log(f'{jobid}_epoch_{epoch}----train acc: {train_acc:.3f}----val acc: {val_acc:.3f}',log_file)
        if epoch > int(params.EPOCH * 0.2):
            if val_acc > best_auc:
                best_auc = val_acc
                best_epoch = epoch
                tsave_dict = {
                    'classifer': Classifier .state_dict()
                }
                torch.save(tsave_dict, save_dir)
                print_log(f' Save best model: ', log_file)
                print_log(f' the current best val_auc: {val_acc}, from epoch {best_epoch}', log_file) 

    print(f'{metric[1] * params.EPOCH / timer.sum():.1f} examples/sec 'f'on {str(params.device)}')
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_training_loss) + 1), avg_training_loss, label='Training Loss')
    plt.plot(range(1, len(avg_val_loss) + 1), avg_val_loss, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2.5) 
    plt.xlim(0, len(avg_training_loss) + 1)  
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig('result/%s_plot.png' % jobid, bbox_inches='tight')
    print('Training finished!')


def model_train(classifier, connection, num_samples, train_loader, criterion,optimizer, params, f_log):
    training_loss = []
    timer = Timer()
    metric = Accumulator(2) 
    classifier.train()
    print_log(f'Current classifier learning rate is {optimizer.param_groups[0]["lr"]}', f_log)

    for i, batch in enumerate(train_loader, 0):
        patches, labels = batch
        labels = labels.to(params.device)  
        patches = patches.to(params.device)  
        aggregated_features_list = []
        for j in range(num_samples):
            aggregated_features = connection(patches[j])
            aggregated_features_list.append(aggregated_features)

        aggregated_features_batch = torch.stack(aggregated_features_list, dim=0)  
        outputs = classifier(aggregated_features_batch)
        loss = criterion(outputs, labels)
        training_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        metric.add(accuracy(_mode=False, output=outputs, target=labels), num_samples)
        timer.stop()
    return training_loss, metric, timer


def bag_model_validation(classifier, connection, val_loader, criterion, params, f_log):
    val_loss = []  
    metric2 = Accumulator(2)
    classifier.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader, 0):
            patches, labels = batch
            labels = labels.to(params.device)  
            patches = patches.to(params.device)  
            aggregated_features_list = []
            for j in range(params.num_samples):
                aggregated_features = connection(patches[j])
                aggregated_features_list.append(aggregated_features)

            aggregated_features_batch = torch.stack(aggregated_features_list, dim=0)
            outputs = classifier(aggregated_features_batch)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            metric2.add(accuracy(_mode=False, output=outputs, target=labels), params.num_samples)

    return val_loss, metric2

if __name__ == "__main__":
    main()
