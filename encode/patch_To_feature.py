import argparse   
import os
from torch.utils.data import Dataset
import torchvision.transforms as T   
import pickle        
from tool.resnet50_pretrained_TCGA import *   
import numpy as np
from encode.dataloader_postcut import *


parser = argparse.ArgumentParser(description='encode_patch')
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=1, type=int)  
parser.add_argument('--class_num', default=4, type=int)   


def extract_save_features(extractor, loader, params):
    extractor.eval()  
    with torch.no_grad():
        for idx, batchdata in enumerate(loader):  
            samples = batchdata['patches'].to(params.device)   
            flattened_patches = samples.view(-1, 3, 224, 224)
            slide_path = batchdata['slice_path']    
            slide_path_str = os.path.join(*slide_path) 
            img_name = batchdata['img_name']
            img_name_str = str(img_name[0])  
            print("one batchsize’s patch is listed")
            feat = extractor(flattened_patches)  
            feat_np = feat.cpu().data.numpy()    
            feats = np.array(feat_np)
            np.save(slide_path_str + '\\'+img_name_str+'.npy', feats) 
    return 0


def main( ):
    params = parser.parse_args()
    class_cate = ['0', '1', '2', '3']
    feat_extractor = resnet50(pretrained=True).to(params.device)  #the length of the ouput fetures is 2048
    print(feat_extractor)
    train_loader, test_loader = load_data(class_cate, params.class_num, params.batch_size)
    extract_save_features(feat_extractor, train_loader, params)
    extract_save_features(feat_extractor, test_loader, params)


if __name__ == '__main__':
    main()

