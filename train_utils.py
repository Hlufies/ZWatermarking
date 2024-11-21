import torch
from utils import *
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def Expand_batch(opt, logger, split_rate, train_dataset, val_dataset, ep): 
    try:
        n = (ep+1)/4 
        bs = opt.batch_size + n*32
       
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        valid_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

        logger.info(f'stage7: Training -> batch size: {opt.batch_size}')
    except Exception as e:
        logger.error(f"Expand_batch error! : {e}") 
        exit(-1)
    return train_loader, valid_loader
def LearningRate(opt, ep, lr, logger, optim):
    try:
        if ep+2 % opt.save_per == 0:
            lr *= 0.5 * (1.0 + math.cos(math.pi * ep/opt.save_per / opt.n_epoch))
            optim.param_groups[0]['lr'] = lr
        logger.info(f'stage7: Training -> Learning Rate: epoch {ep}, lr {lr}')
    except Exception as e:
        logger.error(f"LearningRate error! : {e}") 
        exit(-1)
    return lr
def updata_model_k(Z_Model, model_k):
    m = 0.999
    for param_q, param_k in zip(
        Z_Model.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q * (1-m)
        param_k.requires_grad = False
    return Z_Model, model_k
def get_label_indice(labels):
    label_indices = {}

    # 遍历标签张量，记录每个标签对应的索引
    for i, label in enumerate(labels):
        if label.item() not in label_indices.keys():
            label_indices[label.item()] = []
        label_indices[label.item()].append(i)
    return label_indices
def get_negative_samples(label_indices, queue, key, data):
    # 找出负样本 
    label_indices_set = set()
    for ky, ids in label_indices.items():
        if ky == key:
            continue
        label_indices_set.update(ids)  # 将标签索引列表中的索引添加到集合中
    # 负样本 
    negative_indices = label_indices_set 
    # 表示在batch负样本是否为空
    flag_negative = False 
    if not negative_indices:
        flag_negative = True
    else:
        negative_indices = list(negative_indices)
    # 如果batch种存在负样本
    if negative_indices is not [] and flag_negative == False:
        negative_indices = torch.tensor(negative_indices, dtype=torch.int)
        negative_samples = data[negative_indices]
        negative_samples = torch.cat((negative_samples, queue), dim=0)
    else:
        negative_samples = queue
    return negative_samples
def train_stage1(opt, Z_Model, data_clone, features_clone, zws, wm, labels, class_labels_clone, scaler, optim):
    positive_indices = torch.nonzero(labels.int()).squeeze(-1)
    # 水印解码器 watermark decoder

    wm_logits, _ = Z_Model.w_decoder(data_clone, features_clone, zws)
    wm_positive_logits, _ = Z_Model.w_decoder(data_clone[positive_indices], features_clone[positive_indices], zws[positive_indices])
    loss_wm = torch.nn.MSELoss()(wm_logits, wm.float()) 
    loss_wm_positives = torch.nn.MSELoss()(wm_positive_logits, wm[positive_indices].float())
    # 样本水印鉴别器 discriminator
    wm_classes_logits, _ = Z_Model.mutExtor(data_clone, features_clone, zws)
    loss_discriminator = torch.nn.CrossEntropyLoss()(wm_classes_logits, class_labels_clone ) 
    
    binary_watermark = (wm_logits[positive_indices] >= 0).float()
    binary_watermark_int = torch.round(binary_watermark).long()

    binary_watermark_ori = (wm[positive_indices] >= 0).float()
    binary_watermark_int_ori = torch.round(binary_watermark_ori).long()

    acc = torch.sum(binary_watermark_int==binary_watermark_int_ori) / (binary_watermark_int.shape[0]*binary_watermark_int.shape[-1])
    loss_stage_1 = 0.7*loss_discriminator + 0.2*loss_wm + 0.7* loss_wm_positives
    scaler.scale(loss_stage_1).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(parameters=Z_Model.parameters(), max_norm=opt.grad_clip_norm)
    scaler.step(optim)
    scaler.update()
    return loss_discriminator, loss_wm, loss_wm_positives, loss_stage_1, acc
def train_stage3(opt, Z_Model, RIp, data, zws, labels, scaler, optim):

    data_ori = data.clone()
    features, data = Z_Model.domainEncoder(data)
    positive_indices = torch.nonzero(labels.int()).squeeze(-1)
    negative_indices = torch.nonzero(labels == 0).squeeze(-1)
    # 水印解码器 watermark decoder
    out, _ = Z_Model.w_decoder(data, features, zws)
    ori, random1 = RIp(out[positive_indices],features[positive_indices], zws[positive_indices])
    random2, random3 = RIp(out[negative_indices],features[negative_indices], zws[negative_indices])

    loss1 = nn.MSELoss()(ori, data_ori[positive_indices])
    
    r1 = torch.full((random1.shape[0], 4, 64, 64), 0.005).to(Z_Model.device)
    # loss2 = nn.MSELoss()(random1 , data_ori[positive_indices])
    loss2 = nn.MSELoss()(random1 , r1).to(Z_Model.device)

    r2 = torch.full((random2.shape[0], 4, 64, 64), 0.005).to(Z_Model.device)
    r3 = torch.full((random3.shape[0], 4, 64, 64), 0.005).to(Z_Model.device)
    loss3 = nn.MSELoss()(random2 , r2)
    loss4 = nn.MSELoss()(random3 , r3)
    # loss3 = nn.MSELoss()(random2, data_ori[negative_indices])
    # loss4 = nn.MSELoss()(random3, data_ori[negative_indices])
    loss = loss1 + loss2 + loss3 + loss4
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(parameters=Z_Model.parameters(), max_norm=opt.grad_clip_norm)
    scaler.step(optim)
    scaler.update()
    return loss
def Contrasive_Learning_psi_1(positive_data, positive_samples, negative_samples):
    positive_data = positive_data.expand(positive_samples.shape[0], -1)
    l_pos = torch.einsum("nc,nc->n", [positive_data, positive_samples]).unsqueeze(-1)
    transposed_negative_samples = torch.transpose(negative_samples, 0, 1)
    l_neg = torch.einsum("nc,ck->nk", [positive_data, transposed_negative_samples])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /=(0.07*2)
    labels_logits = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
    loss = torch.nn.CrossEntropyLoss()(logits, labels_logits)
    return loss
def get_pos_and_neg_of_samples(positive_pos, positive_data, positive_samples):
    
                    
    zero_indices = torch.eq(positive_pos, 0)   
    one_indices = torch.eq(positive_pos, 1)
    nonzero_indices = torch.nonzero(zero_indices).squeeze()
    
    
    if nonzero_indices.size() == torch.Size([]) and nonzero_indices.numel()==1:
        nonzero_indices = torch.tensor([nonzero_indices.item()], device=nonzero_indices.device)
    nonone_indices = torch.nonzero(one_indices).squeeze()
    if nonone_indices.size() == torch.Size([]) and nonone_indices.numel()==1:
        nonone_indices = torch.tensor([nonone_indices.item()], device=nonone_indices.device)

    nonzero_flag = False
    nonone_flag = False
    positive_data = positive_data[0].unsqueeze(0)
    
    samples_pos, samples_neg = None, None
    if nonzero_indices.numel() != 0:
        samples_neg = positive_samples[nonzero_indices]
    else:
        nonzero_flag = True
    
    if nonone_indices.numel() != 0:
        samples_pos = positive_samples[nonone_indices]
    else:
        samples_pos = positive_data
        nonone_flag = True
    '''
        nonzero_flag = True, 表示间类只有正样例
        nonone_flag = True， 表示间类有负样例和正样例
    '''
    return samples_pos, samples_neg, positive_data, nonone_flag, nonzero_flag
def Contrasive_Learning_psi_2(samples_pos, samples_neg, positive_data, nonone_flag, nonzero_flag, positive_pos):
    if nonzero_flag == True:
        s_loss = torch.nn.CrossEntropyLoss()(positive_pos.to(torch.float), torch.ones_like(positive_pos).to(torch.float).to(positive_pos.device))
    else: 
        positive_data = positive_data.expand(samples_pos.shape[0], -1)
        # print(f'positive_data:{positive_data.shape}')
        s_pos = torch.einsum("nc,nc->n", [positive_data, samples_pos]).unsqueeze(-1)
        s_neg = torch.einsum("nc,ck->nk", [positive_data, torch.transpose(samples_neg, 0, 1)])
        s_logits = torch.cat([s_pos, 0.6*s_neg], dim=1)
        s_logits /=(0.7*2)
        s_labels_logits = torch.zeros(s_logits.shape[0], dtype=torch.long).to(s_logits.device)
        s_loss = torch.nn.CrossEntropyLoss()(s_logits, s_labels_logits)
    return s_loss
def repeat_tensor(BC, anchor_domain, anchor_domain_feature, anchor_zws):
    if BC > 1:
        return anchor_domain, anchor_domain_feature, anchor_zws
    pass
def discriminator_loss_1(Z_Model, data_clone, features_clone, class_labels_clone, zws):
    wm_classes_logits, _ = Z_Model.mutExtor(data_clone, features_clone, zws)
    loss_wm_class = torch.nn.CrossEntropyLoss()(wm_classes_logits, class_labels_clone) 
    return loss_wm_class
def wm_extractor_loss(Z_Model, positive_samples, positive_features, positive_zws, positive_wm, anchor_domain, anchor_domain_feature, anchor_zws):

    if positive_samples.shape[0] == 1:
        BC = 10
    else:
        BC = positive_samples.shape[0]
    anchor_domain_repeat = anchor_domain.repeat(BC, 1)
    anchor_domain_feature_repeat = anchor_domain_feature.repeat(BC, 1, 1, 1)
    anchor_zws_repeat = anchor_zws.repeat(BC, 1)


    _, anchor_perceptual_WmDecoder = Z_Model.w_decoder(anchor_domain_repeat, anchor_domain_feature_repeat, anchor_zws_repeat)



    if positive_samples.shape[0] == 1:
        positive_samples_ = positive_samples.expand(positive_samples.shape[0]*10, -1)
        positive_features_ = positive_features.expand(positive_features.shape[0]*10, -1,-1,-1)
        positive_zws_ = positive_zws.expand(positive_zws.shape[0]*10, -1)
        watermark_logits, watermark_perceptual = Z_Model.w_decoder(positive_samples_, positive_features_, positive_zws_)
        positive_wm = positive_wm.expand(positive_samples.shape[0]*10, -1)
    else:
        watermark_logits, watermark_perceptual = Z_Model.w_decoder(positive_samples, positive_features, positive_zws)
    loss_wm = torch.nn.MSELoss()(watermark_logits, positive_wm.float()) 

    Perceptual_loss = torch.nn.MSELoss()(watermark_perceptual,anchor_perceptual_WmDecoder)
    return loss_wm, Perceptual_loss
def discriminator_loss_2(Z_Model, data, features, disc, zws):
    disc_samples = data.clone()
    disc_features = features.clone()
    disc_labels = disc.clone()
    disc_zws = zws.clone()
    disc_classes_logits, _ = Z_Model.discriminator(disc_samples, disc_features, disc_zws)
    
    loss_disc = torch.nn.CrossEntropyLoss()(disc_classes_logits, disc_labels)
    return loss_disc
