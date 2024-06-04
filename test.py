import os
import random
import numpy as np
import argparse
import json
import time
import torch.nn.functional as F
from encode.dataloader_feature import load_data_feature_test_wsi, load_data_feature
import torch
from tool.model_1024 import ps_vit_1024_16head
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tool.tool_ac import Timer, Accumulator, accuracy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


parser = argparse.ArgumentParser(description='PSWS_TEST/VAL')
parser.add_argument('--name', default='PSWS_TEST/VAL', type=str)
parser.add_argument('--EPOCH', default=1, type=int)  
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--log_dir', default='debug_log', type=str)   
parser.add_argument('--class_num', default=4, type=int)   

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

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)  
    log_dir = os.path.join(params.log_dir, 'log.txt')  
    save_dir = os.path.join(params.log_dir, 'Optimal Model.pth')  

    z = vars(params).copy()  
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))  
    log_file = open(log_dir, 'a')  

    class_cate = ['0', '1', '2', '3']
    class_number = params.class_num
    slice_val_loader = load_data_feature_test_wsi()   

    Classifier = ps_vit_1024_16head()
    print(Classifier)
    checkpoint = torch.load(save_dir)
    Classifier.load_state_dict(checkpoint['classifer'])
    Classifier.to(params.device)
    Connection = ConnectionModule().to(params.device)

    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean').to(params.device) 

    for epoch in range(params.EPOCH):
        print_log(f'>>>>>>>>>>> Testing: {epoch}', log_file)  

        model_test_slice(classifier=Classifier, connection=Connection,
                               val_loader=slice_val_loader,
                               criterion1=ce_loss, params=params, f_log=log_file)

    print('Testing finished!')


def model_test_slice(classifier, connection, val_loader, criterion1, params, f_log):
    classifier.eval()
    val_loss = []
    metric2 = Accumulator(2)
    whole_label = []
    whole_prediction = []
    inference_times = []  
    with torch.no_grad():
        for i, batch in enumerate(val_loader, 0):
            print("one slice started")
            start_time = time.time()  
            wsi_labels = []
            patches, label, path = batch
            label = label.to(params.device)
            label_item = label.item()
            whole_label.append(label_item)
            patches = patches.squeeze(0)
            patches = patches.to(params.device)
            num_big_patch = patches.shape[0]
            wsi_labels.extend([label_item] * num_big_patch)
            wsi_labels = torch.tensor(wsi_labels)
            wsi_labels = wsi_labels.to(params.device)
            aggregated_features_list = []
            for j in range(num_big_patch):
                aggregated_features = connection(patches[j])
                aggregated_features_list.append(aggregated_features)
            aggregated_features_batch = torch.stack(aggregated_features_list, dim=0)
            outputs = classifier(aggregated_features_batch)  
            output_softmax = F.softmax(outputs, dim=1)  
            aggregated_probs = torch.mean(output_softmax, dim=0)  
            predicted_label_final = torch.argmax(aggregated_probs)  
            wsi_prediction = predicted_label_final.item()  
            print(path)
            print(wsi_prediction)
            print(label)
            inference_time = time.time() - start_time  
            inference_times.append(inference_time)
            whole_prediction.append(wsi_prediction)
            loss = criterion1(outputs, wsi_labels)  
            val_loss.append(loss.item())
            if wsi_prediction == label:
                metric2.add(1, 1)
            else:
                metric2.add(0, 1)
        whole_label = np.array(whole_label)
        whole_prediction = np.array(whole_prediction)
        val_acc = metric2[0] / metric2[1]
        average_loss = np.average(val_loss)

        average_inference_time = sum(inference_times) / len(inference_times)
        print(f"Average inference time: {average_inference_time:.6f} s")

        precision, recall, f1, _ = precision_recall_fscore_support(whole_label, whole_prediction, average='macro')
        print(
            f"WSI-level index：acc={val_acc:.6f}，recall={recall:.6f}，precision={precision:.6f}，F1={f1:.6f}，avg_loss={average_loss:.4f}")

        conf_matrix = confusion_matrix(whole_label, whole_prediction)
        class_labels = sorted(np.unique(whole_label))
        conf_matrix_labeled = np.vstack((
            np.array([[" "] + [f"P{label}" for label in class_labels]], dtype=object),
            np.column_stack(
                (np.array([f"T{label}" for label in class_labels], dtype=object), conf_matrix)
            )
        ))
        print("Labeled Confusion Matrix:")
        for row in conf_matrix_labeled:
            print(" ".join(f"{item:8}" for item in row))

        class_name = ["Leiomyosarcoma", "Synovial sarcoma", "Undifferentiated sarcoma", "Liposarcoma"]
        classification_rep = classification_report(whole_label, whole_prediction, target_names=class_name)
        print("Classification Report:")
        print(classification_rep)

        y_true_one_hot = label_binarize(whole_label, classes=range(params.class_num))
        y_score_one_hot = label_binarize(whole_prediction, classes=range(params.class_num))
        # Micro-average AUC
        micro_fpr, micro_tpr, _ = roc_curve(y_true_one_hot.ravel(), y_score_one_hot.ravel())
        micro_roc_auc = auc(micro_fpr, micro_tpr)
        print(f'Micro-average AUC: {micro_roc_auc:.6f}')


def model_test_big_patch(classifier, connection, val_loader, criterion1, params, f_log):
    classifier.eval()
    val_loss = []
    metric2 = Accumulator(2)
    with torch.no_grad():
        for i, batch in enumerate(val_loader, 0):
            wsi_labels = []
            patches, label, path = batch
            label_item = label.item()
            patches = patches.squeeze(0)
            patches = patches.to(params.device)
            num_big_patch = patches.shape[0]
            wsi_labels.extend([label_item] * num_big_patch)
            wsi_labels = torch.tensor(wsi_labels)
            wsi_labels = wsi_labels.to(params.device)
            aggregated_features_list = []
            for j in range(num_big_patch):
                aggregated_features = connection(patches[j])
                aggregated_features_list.append(aggregated_features)
            aggregated_features_batch = torch.stack(aggregated_features_list, dim=0)
            outputs = classifier(aggregated_features_batch)   
            loss = criterion1(outputs, wsi_labels)  
            val_loss.append(loss.item())
            metric2.add(accuracy(_mode=False, output=outputs, target=wsi_labels), num_big_patch)
        val_acc = metric2[0] / metric2[1]
        average_loss = np.average(val_loss)
        print(
            f"Bag_lavel index：acc={val_acc:.6f}，avg_loss={average_loss:.4f}")


if __name__ == "__main__":
    main()
