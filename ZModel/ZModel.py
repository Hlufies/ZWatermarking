import torch.nn as nn
from torchvision.models import resnet
import torch
import os
import torch.nn.init as init



class DomainEncoder(nn.Module):
    """
    An encoder network (image -> feature_dim)
    """
    def __init__(self, arch, feature_dim, cifar_small_image=False, artworks='', device=torch.device('cpu')):
        super(DomainEncoder, self).__init__()
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim)
        if artworks=='artworks':
            # print("This is artworks config")
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
        
        self.domainFeatures = []
        self.domain= []
        layers_nums = 0
        for name, module in net.named_children():
            layers_nums += 1
           
            if isinstance(module, nn.Linear):
                if layers_nums <= 8:
                    self.domainFeatures.append(nn.Flatten(1))
                    self.domainFeatures.append(module)
                else:
                    self.domain.append(nn.Flatten(1))
                    self.domain.append(module)
            else:
                if cifar_small_image:
                    # replace first conv from 7x7 to 3x3
                    if name == 'conv1':
                        module = nn.Conv2d(module.in_channels, module.out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=False)
                    # drop first maxpooling
                    if isinstance(module, nn.MaxPool2d):
                        continue
                if layers_nums <= 8:
                    self.domainFeatures.append(module)
                else:
                    self.domain.append(module)
        self.domainFeatures = nn.Sequential(*self.domainFeatures).to(device)
        self.domain = nn.Sequential(*self.domain).to(device)

    def forward(self, x):
        x_features = self.domainFeatures(x)
        x_domain = self.domain(x_features)
        # torch.Size([B, 512, 8, 8])
        # torch.Size([B, 128])
        # domain_dim = 128
        return x_features, x_domain

class MutExtor(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(MutExtor, self).__init__()  # 修复这一行
        out_channels +=1 # 加一个是异常样本
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_Sequential = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.Features_fusion = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )
        self.Features_reduce = nn.Sequential(
            nn.Linear(in_features=4*in_channels, out_features=2*in_channels),
            nn.BatchNorm1d(256)
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Linear(in_features=in_channels, out_features=out_channels)
        )
        self.fc_z = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.fc_f = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.Adapt = nn.AdaptiveAvgPool2d(1)

    def forward(self, z, f, own=None):
        
        '''
        params:
        z shape:   [batch_size, in_channels]
        f shape:   [batch_size, in_channels*4, 8, 8]
        own shape: [batch_size, 768]
        out shape: [batch_size, in_channels] -> wm_logtis
        ''' 
        f=self.Conv_Sequential(f)                            
        f = self.Adapt(f).view(f.shape[0],f.shape[1])       # [batch_size, 4*in_channels, 1, 1]
        f = torch.cat((self.Features_reduce(f), f), dim=-1) # [batch_size, 2*in_channels] + [batch, 4*in_channels] -> [batch_size, 4*in_channels]
        f = torch.add(f,own)                                # [batch_size, 4*in_channels]
        z_Feature = f.clone()
        f = self.Features_fusion(f)                         # [batch_size, in_channels]
        out = self.fc_z(z) + self.fc_f(f) + z + f           # [batch_size, in_channels]
        out = self.out(out)                                 # [batch_size, out_channels]
        return out, out
    
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(Discriminator, self).__init__()  # 修复这一行
        # out_channels +=1 # 最后一个是异常样本
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_Sequential = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.Features_fusion = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )
        self.Features_reduce = nn.Sequential(
            nn.Linear(in_features=4*in_channels, out_features=2*in_channels),
            nn.BatchNorm1d(256)
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Linear(in_features=in_channels, out_features=out_channels)
        )
        self.fc_z = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.fc_f = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.Adapt = nn.AdaptiveAvgPool2d(1)


    def forward(self, z, f, own=None):
        '''
        params:
        z shape:   [batch_size, in_channels]
        f shape:   [batch_size, in_channels*4, 8, 8]
        own shape: [batch_size, 768]
        out shape: [batch_size, in_channels] -> wm_logits
        '''

        f=self.Conv_Sequential(f)                            
        f = self.Adapt(f).view(f.shape[0],f.shape[1])       # [batch_size, 4*in_channels, 1, 1]
        f = torch.cat((self.Features_reduce(f), f), dim=-1) # [batch_size, 2*in_channels] + [batch, 4*in_channels] -> [batch_size, 4*in_channels]
        f = torch.add(f,own)                                # [batch_size, 4*in_channels]
        f = self.Features_fusion(f)                         # [batch_size, in_channels]
        z_Feature = f.clone()
        out = self.fc_z(z) + self.fc_f(f) + z + f           # [batch_size, in_channels]
        out = self.out(out)                                 # [batch_size, out_channels]
        return out, out

class WmDecoder(nn.Module):
    def __init__(self,in_channels, out_channels, device):
        super(WmDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_Sequential = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.Features_fusion = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Linear(in_features=768, out_features=128),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )
        self.Features_reduce = nn.Sequential(
            nn.Linear(in_features=4*in_channels, out_features=2*in_channels),
            nn.BatchNorm1d(256)
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Linear(in_features=in_channels, out_features=out_channels)
        )
        self.fc_z = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.fc_f = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.Adapt = nn.AdaptiveAvgPool2d(1)
    def forward(self, z, f, own=None):
        '''
        params:
        z shape:   [batch_size, in_channels]
        f shape:   [batch_size, in_channels*4, 8, 8]
        own shape: [batch_size, 768]
        out shape: [batch_size, in_channels] -> wm_logits
        '''
        
        f=self.Conv_Sequential(f)

        f = self.Adapt(f).view(f.shape[0],f.shape[1])       # [batch_size, 4*in_channels, 1, 1]
        f = torch.cat((self.Features_reduce(f), f), dim=-1) # [batch_size, 2*in_channels] + [batch, 4*in_channels] -> [batch_size, 4*in_channels]
        f = torch.add(f,own)                                # [batch_size, 4*in_channels]
        f = self.Features_fusion(f)                         # [batch_size, in_channels]
        z_Feature = f.clone()
        out = self.fc_z(z) + self.fc_f(f) + z + f           # [batch_size, in_channels]
        out = self.out(out)                                 # [batch_size, out_channels]
        return out, out

class ZModel(nn.Module):
    def __init__(self, opt, pretrain_path=None, soda=None, pretrained_soda_or_encoder=False, device=torch.device('cpu')):
        super(ZModel, self).__init__()
        self.opt = opt
        self.device = device
   
        self.bit_len = opt.bit_len
        self.discriminator = Discriminator(opt.domain_dim, 2, device)
        self.w_decoder = WmDecoder(opt.domain_dim,self.bit_len, device)
        self.domainEncoder = DomainEncoder(**opt.encoder).to(device)
        self.mutExtor = MutExtor(opt.domain_dim, opt.protectedUnits_nums,device).to(device)
        
        if pretrained_soda_or_encoder == True and os.path.exists(pretrain_path):
            self.load_features_watermark_model_encoder(pretrain_path, device)     
        elif pretrained_soda_or_encoder == False and soda is not None:
            self.load_features_watermark_model_soda(soda)

        # create the queue
        self.register_buffer("queue", torch.randn(256000, 128))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels", torch.zeros(3, 256000))
        self.T = 256000
    
    def _dequeue_and_enqueue(self, keys_1, keys_2):
        # gather keys before updating queue
        batch_size = keys_1.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.T % batch_size == 0  # for simplicity
        if self.T - ptr - 1 < batch_size:
            batch_size_1 = self.T - ptr 
            self.queue[ptr : ptr + batch_size_1,:] = keys_1[:batch_size_1, :]
            self.queue_labels[:, ptr : ptr + batch_size_1] = keys_2[:, : batch_size_1]
            batch_size -= batch_size_1
            ptr = (ptr + batch_size_1)%self.T
            keys_1 = keys_1[batch_size_1 :, : ]
            keys_2 = keys_2[:, batch_size_1:]
           
            assert keys_1.shape[0] == batch_size
       
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size,:] = keys_1
        self.queue_labels[:, ptr : ptr + batch_size] = keys_2
        ptr = (ptr + batch_size) % self.T  # move pointer

        self.queue_ptr[0] = ptr
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def load_features_watermark_model_encoder(self,pretrain_path, device):
        checkpoint = torch.load(pretrain_path, map_location=device)
        self.domainEncoder.load_state_dict(checkpoint)
        print(f'loading encoder from encoder.pth successfully!')
    def load_features_watermark_model_soda(self, soda):
        encoder = soda.encoder
        # 遍历 encoder 模型的参数并赋值给 DomainEncoder 模型
        encoder_params = encoder.named_parameters()
        watermark_params = self.domainEncoder.named_parameters()
        encoder_params_length = sum(1 for _ in encoder_params)
        watermark_params_length = sum(1 for _ in watermark_params)
        assert encoder_params_length == watermark_params_length # "Encoder and watermark models have different numbers of parameters"
        print("Loading ==================================================")
        encoder_params = encoder.named_parameters()
        watermark_params = self.domainEncoder.named_parameters()
        for encoder_param, watermark_param in zip(encoder_params, watermark_params):    
            watermark_param[1].data.copy_(encoder_param[1].data)
        print(f'loading encoder from soda.pthsuccessfully!')
    def compute_features_similarity(self, latents_artwork, latents_generation, diff_pix, diff_latent):
        # assert latents_artwork.device == latents_generation.device == diff_pix.device == diff_latent.device
        assert latents_artwork.shape[-1] == latents_generation.shape[-1]
        assert latents_generation.shape[0] == diff_pix.shape[0] and diff_pix.shape == diff_latent.shape
        latents_artwork_ = nn.Sigmoid()(latents_artwork)
        latents_generation_ = nn.Sigmoid()(latents_generation)
        cos_similarities = torch.nn.functional.cosine_similarity(latents_artwork_, latents_generation_, dim=1)
        similar_indices_1 = torch.nonzero(cos_similarities > 0.9975)
        # find the pix error 
        diff_pix_mean = torch.mean(diff_pix)
        diff_pix_error = diff_pix - diff_pix_mean
        similar_indices_2 = torch.nonzero(diff_pix_error <= 0)
        # find the latent error 
        diff_latent_mean = torch.mean(diff_latent)
        diff_latent_error = diff_latent - diff_latent_mean
        similar_indices_3 = torch.nonzero(diff_latent_error <= 0)

       
        # 处理 similar_indices_1
        if similar_indices_1.numel() == 1:  # 检查张量中元素的总数是否为1
            similar_indices_1 = similar_indices_1.item()  # 获取张量中的单个值
            set_1 = set([similar_indices_1])
        else:
            set_1 = set(similar_indices_1.squeeze(-1).tolist())
        # 处理 similar_indices_2
        if similar_indices_2.numel() == 1:  # 检查张量中元素的总数是否为1
            similar_indices_2 = similar_indices_2.item()  # 获取张量中的单个值
            set_2 = set([similar_indices_2])
        else:
            set_2 = set(similar_indices_2.squeeze(-1).tolist())

        # 处理 similar_indices_3
        if similar_indices_3.numel() == 1:  # 检查张量中元素的总数是否为1
            similar_indices_3 = similar_indices_3.item()  # 获取张量中的单个值
            set_3 = set([similar_indices_3])
        else:
            set_3 = set(similar_indices_3.squeeze(-1).tolist())



        # 计算三个集合的交集
        intersection_all = list(set_1.intersection(set_2, set_3))
        similar_indices = torch.tensor(intersection_all)
        all_indices = torch.arange(latents_generation.shape[0]).to(latents_artwork.device)
        un_similar_indices = torch.masked_select(all_indices, torch.logical_not(torch.isin(all_indices, similar_indices)))
        return similar_indices, un_similar_indices
    def compute_z_similarity(self, artwork_watermarks, labels, z):
        if labels[0].item() == 0:
            return labels
        pos_or_neg = torch.zeros_like(labels)
        for idx, (label, z_z) in enumerate(zip(labels, z)):
            
            a_w = artwork_watermarks[0]
            # 计算欧氏距离
            euclidean_distance = torch.sqrt(torch.sum((z_z - a_w) ** 2))
            # 计算余弦相似度
            cosine_similarity = torch.nn.functional.cosine_similarity(z_z, a_w)
            # watermark distribution margins = 1
            if euclidean_distance <= 1 and cosine_similarity >= 0.999:
                pos_or_neg[idx] = 1
                
        return pos_or_neg
    def watermark_bit(self, latent):
        with torch.no_grad():
            _, watermark = self.domainEncoder(latent)
        return watermark
    def gpu_model(self, opt=None, flags=True):
        if flags==False:
            self.device= opt.device
      
        self.domainEncoder = self.domainEncoder.to(self.device)
        self.mutExtor  =self.mutExtor.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.w_decoder = self.w_decoder.to(self.device)
        

   
    def forward(self):
       pass
