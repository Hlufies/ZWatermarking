import torch.nn as nn
from torchvision.models import resnet
class Network(nn.Module):
    """
    An encoder network (image -> feature_dim)
    """
    def __init__(self, arch, feature_dim, cifar_small_image=False, IP='IP'):
        super(Network, self).__init__()
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim)
        
        conv1_weight = net.conv1.weight
        out_channels = conv1_weight.size(0)

        # 构建新的卷积层参数字典
        conv2_params = {
            "in_channels": 4,  # 修改输入通道数为4
            "out_channels": out_channels,
            "kernel_size": conv1_weight.size()[2:],  # 保持原始卷积核大小
            "stride": net.conv1.stride,
            "padding": net.conv1.padding,
            "bias": net.conv1.bias is not None  # 检查是否有偏置项
        }
        net.conv1 = nn.Conv2d(**conv2_params)
        
        self.encoder = []
        # self.encoder_features = []
        layers_nums = 0
        for name, module in net.named_children():
            layers_nums += 1      
            if isinstance(module, nn.Linear):
                self.encoder.append(nn.Flatten(1))
                self.encoder.append(module)
            else:
                if cifar_small_image:
                    # replace first conv from 7x7 to 3x3
                    if name == 'conv1':
                        module = nn.Conv2d(module.in_channels, module.out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=False)
                    # drop first maxpooling
                    if isinstance(module, nn.MaxPool2d):
                        continue
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)
    def forward(self, x):
        return self.encoder(x)
