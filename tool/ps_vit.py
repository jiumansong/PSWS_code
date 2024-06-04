import torch
import torch.nn as nn
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from layers import ProgressiveSample
from tool.transformer_block import TransformerEncoderLayer
import torch.nn.functional as F
import torch.nn.init as init


class CustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomClassifier, self).__init__()

        self.fc1 = nn.Linear(16384, 1024)  
        self.norm1 = nn.LayerNorm(1024)  
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.norm2 = nn.LayerNorm(256) 
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        length = x.shape[1] * x.shape[2]
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = self.norm1(x) 
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x) 
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class PSViTLayer(nn.Module):
    def __init__(self,
                 feat_size,   
                 dim,
                 num_heads,   
                 mlp_ratio=4.,   
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,    
                 norm_layer=nn.LayerNorm,
                 position_layer=None,
                 pred_offset=True,
                 gamma=0.1,     
                 offset_bias=False):
        super().__init__()

        self.feat_size = float(feat_size)

        self.transformer_layer = TransformerEncoderLayer(dim,
                                                         num_heads,
                                                         mlp_ratio,
                                                         qkv_bias,
                                                         qk_scale,
                                                         drop,
                                                         attn_drop,
                                                         drop_path,
                                                         act_layer,
                                                         norm_layer)
        self.sampler = ProgressiveSample(gamma)   

        self.position_layer = position_layer      
        if self.position_layer is None:
            self.position_layer = nn.Linear(2, dim)  

        self.offset_layer = None
        if pred_offset:
            self.offset_layer = nn.Linear(dim, 2, bias=offset_bias)  

    def reset_offset_weight(self):
        if self.offset_layer is None:
            return
        nn.init.constant_(self.offset_layer.weight, 0)  
        if self.offset_layer.bias is not None:
            nn.init.constant_(self.offset_layer.bias, 0)   

    def forward(self,
                x,
                point,
                offset=None,
                pre_out=None):

        if offset is None:
            offset = torch.zeros_like(point)  

        sample_feat = self.sampler(x, point, offset)   
        sample_point = point + offset.detach()   
        pos_feat = self.position_layer(sample_point / self.feat_size)  
        attn_feat = sample_feat + pos_feat   
        if pre_out is not None:  
            attn_feat = attn_feat + pre_out

        attn_feat = self.transformer_layer(attn_feat)
        out_offset = None
        if self.offset_layer is not None:
            out_offset = self.offset_layer(attn_feat)   #

        return attn_feat, out_offset, sample_point, sample_feat, pos_feat   


class PSViT(nn.Module):
    def __init__(self,
                 num_point_w=14,   
                 num_point_h=14,
                 num_classes=0,    
                 num_iters=4,    
                 embed_dim=192,  
                 num_heads=12,   
                 mlp_ratio=4.,   
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,   
                 stem_layer=None,
                 offset_gamma=0.1,
                 offset_bias=False):    
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        assert num_iters >= 1   
        self.feat_size = 9   
        self.num_point_w = num_point_w
        self.num_point_h = num_point_h
        self.register_buffer('point_coord', self._get_initial_point())   
        self.pos_layer = nn.Linear(2, self.embed_dim)   
        self.stem = stem_layer   
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_iters)]
        self.ps_layers = nn.ModuleList()   
        for i in range(num_iters):    
            self.ps_layers.append(PSViTLayer(feat_size=self.feat_size,
                                             dim=self.embed_dim,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop_rate,
                                             attn_drop=attn_drop_rate,  
                                             drop_path=dpr[i],    
                                             norm_layer=norm_layer,
                                             position_layer=self.pos_layer,
                                             pred_offset=i < num_iters - 1,  
                                             gamma=offset_gamma,
                                             offset_bias=offset_bias))

        self.CustomClassifier = CustomClassifier(num_classes=self.num_classes)
        self.apply(self._init_weights)   
        for layer in self.ps_layers:   
            layer.reset_offset_weight()    

        self.transformer_layer_classify = TransformerEncoderLayer(dim=self.embed_dim,
                                                                  num_heads=num_heads,
                                                                  mlp_ratio=mlp_ratio,
                                                                  qkv_bias=qkv_bias,
                                                                  qk_scale=qk_scale,
                                                                  drop=drop_rate,
                                                                  attn_drop=attn_drop_rate,
                                                                  drop_path=dpr[i],
                                                                  act_layer=nn.GELU,
                                                                  norm_layer=norm_layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  
            trunc_normal_(m.weight, std=.02)    
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)    
        elif isinstance(m, nn.LayerNorm):     
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_initial_point(self):
        patch_size_w = self.feat_size / self.num_point_w
        patch_size_h = self.feat_size / self.num_point_h
        coord_w = torch.Tensor(
            [i * patch_size_w for i in range(self.num_point_w)])
        coord_w += patch_size_w / 2
        coord_h = torch.Tensor(
            [i * patch_size_h for i in range(self.num_point_h)])
        coord_h += patch_size_h / 2

        grid_x, grid_y = torch.meshgrid(coord_w, coord_h)
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)
        point_coord = torch.cat([grid_y, grid_x], dim=0)
        point_coord = point_coord.view(2, -1)
        point_coord = point_coord.permute(1, 0).contiguous().unsqueeze(0)

        return point_coord

    def forward(self, x):
        batch_size = x.size(0)    
        point = self.point_coord.repeat(batch_size, 1, 1)  
        x = self.stem(x)   F
        ps_out = None
        offset = None
        for layer in self.ps_layers:    
            ps_out, offset, point, sample_feat, pos_feat = layer(x,
                                                       point,
                                                       offset,
                                                       ps_out)
        out = self.CustomClassifier(ps_out)             

        return out
