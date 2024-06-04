import torch
from tool.ps_vit import PSViT
from tool.pre_model import PreModel, DirectConnectModule


def ps_vit_1024_16head(**kwargs):

    stem = PreModel()    
    model = PSViT(embed_dim=1024,  
                  num_iters=8,     
                  num_point_h=4,  
                  num_point_w=4,
                  num_classes=4,
                  num_heads=16,   
                  drop_rate=0.2,   
                  drop_path_rate=0.,
                  mlp_ratio=3.,
                  stem_layer=stem,
                  offset_gamma=1.0,   
                  offset_bias=True,
                  **kwargs)
    return model


