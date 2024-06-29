import torch
from torchvision.models.resnet import Bottleneck, ResNet
from torchsummary import summary
import torch.nn as nn
import torch
from thop import profile
from thop import clever_format


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  
        x = self.flatten(x)
        return x


"""
def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url
"""


def resnet50(pretrained, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #Mingu Kang."Self-supervised pre-trained weights on TCGA," Github, 2023, https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
        file_path = 'the path of bt_rn50_ep200.torch'
        #the path of mocov2_rn50_ep200.torch'
        #the path of swav_rn50_ep200.torch'
        verbose = model.load_state_dict(
            torch.load(file_path), strict=False
        )
        print(verbose)
    return model



def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    net = resnet50(pretrained=True)
    net = net.to(device)
    print(net)
    summary(net, (3, 224, 224))

test()


