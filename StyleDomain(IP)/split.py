import argparse
import os
from functools import partial

import torch
# import torch.distributed as dist
import yaml
from metric import KNN, LinearProbe
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datetime
from model.SODA import SODA
from model.encoder_back import Network_Features
from model.encoder_back import Network, watermark_decoder
from model.decoder import UNet_decoder
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP, print0
from datasets import get_dataset, WatermarkDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

 
def loss_vision(loss_sum, loss_sum_s, loss_sum_l, loss_sum_w):
    iterations = range(1, len(loss_sum) + 1)
    # 创建图表
    plt.figure(figsize=(25, 12))

    # 绘制总损失曲线
    plt.plot(iterations, loss_sum, label='Total Loss', marker='o')

    # 绘制各个损失的曲线
    plt.plot(iterations, loss_sum_s, label='SimCLR Loss', marker='o')
    plt.plot(iterations, loss_sum_l, label='Label Loss', marker='o')
    plt.plot(iterations, loss_sum_w, label='Watermark Loss', marker='o')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # 显示网格
    plt.grid(True)
    # 保存图像为 PNG 格式
    plt.savefig('loss_curve.png')
def load_soda_state_dict(pretrain_path, device, opt):
    # 加载soda ======================================================
    soda = SODA(encoder=Network(**opt.encoder),
                decoder=UNet_decoder(**opt.decoder),
                **opt.diffusion,
                device=device)
    files = os.listdir(pretrain_path)
    max_epochs = -1
    model_path = ''
    for file in files:
        if 'model' not in file:
            continue
        ep = int(file.split('.')[0].split('_')[-1])
        if ep > max_epochs:
            max_epochs = ep
            model_path = os.path.join(pretrain_path, file)
    print(f'model_path:{model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    # 移除"module"前缀
    new_state_dict = {}
    for key, value in checkpoint['MODEL'].items():
        new_key = key.replace("module.", "")  # 移除"module."前缀
        new_state_dict[new_key] = value
    print(f'new_state_dict:{len(new_state_dict)}')
    soda.load_state_dict(new_state_dict)
    DDP_multiplier = 1
    lr = opt.lrate
    lr *= DDP_multiplier
    optim = get_optimizer([{'params': soda.encoder.parameters(), 'lr': lr * opt.lrate_ratio},
                           {'params': soda.decoder.parameters(), 'lr': lr}], opt, lr=0)
    
    optim.load_state_dict(checkpoint['opt'])
    # 
    return soda, optim
def train(opt):
    yaml_path = opt.config

    use_amp = opt.use_amp

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    tsbd_dir = os.path.join(opt.save_dir, "tensorboard")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = "cuda:1"
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    pretrain_path = current_script_directory+model_dir.split('.')[-1]
    assert os.path.exists(pretrain_path)
    soda, optim = load_soda_state_dict(pretrain_path, device, opt)
    watermark_model = watermark_decoder(opt=opt, soda=soda, pretrained_soda_or_encoder=False).to(device)
    save_path_model = '/newdata/gluo/ZI/Disentangled_Representation_Learning/soda/artworks/ckpts'
    torch.save(watermark_model.state_dict(), os.path.join(save_path_model, f'{opt.artworks_nums}_encoder.pth'))
    # 从优化器的参数组中提取要保存的特定参数
    # selected_params = {key: optim.param_groups[0][key] for key in ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'maximize', 'capturable', 'differentiable', 'fused', 'params']}
    # 将选定的参数保存到文件中
    # torch.save(selected_params, os.path.join(save_path_model, 'optim_selected_params.pth'))
    torch.save(optim.param_groups[0], os.path.join(save_path_model,'optim.pth'))
    exit(0)
    del soda

    artworks_dir = opt.artworks_dir
    samples_dir = opt.samples_dir
    other_path = opt.other_path
    cal_pix_dir = opt.cal_pix_path
    cal_latent_dir = opt.cal_latent_path

    # 按照名称排列
    artwork_name = os.listdir(artworks_dir)
    samples_name = os.listdir(samples_dir)
    cal_pix_name = os.listdir(cal_pix_dir)
    cal_latent_name = os.listdir(cal_latent_dir)
    assert len(artwork_name) == len(samples_name)
    assert len(cal_latent_name) == len(cal_pix_name)
    artwoks_list = []
    samples_list = []
    cal_pix_list = []
    cal_latent_list = []
    for i, a_n in enumerate(artwork_name):
        print(f'the {i+1}th artwork dataset')
        artwork_path = os.path.join(artworks_dir, a_n)
        samples_path = os.path.join(samples_dir, a_n)
        cal_pix_path = os.path.join(cal_pix_dir, a_n)
        cal_latent_path = os.path.join(cal_latent_dir, a_n)
        assert a_n in samples_name and a_n in cal_pix_name and a_n in cal_latent_name
        assert os.path.exists(artwork_path) and os.path.exists(samples_path)
        assert os.path.isfile(artwork_path) and os.path.isfile(samples_path)
        assert os.path.exists(cal_pix_path) and os.path.exists(cal_latent_path)
        assert os.path.isfile(cal_pix_path) and os.path.isfile(cal_latent_path)
        
        artwoks_list.append(torch.load(artwork_path, map_location=torch.device('cpu')))
        samples_list.append(torch.load(samples_path))
        cal_pix_list.append(torch.load(cal_pix_path))
        cal_latent_list.append(torch.load(cal_latent_path))
    
    # First phrase  features -> watermarks
    artworks_watermarks = []
    for a_l in artwoks_list:
        assert a_l.shape[0] == 1 and a_l.shape[1] == 4 and a_l.shape[2] == a_l.shape[3] == 64
        artworks_watermarks.append(watermark_model.watermark_bit(a_l))
    print(f' First phrase has been done!')
    exit(-1)
    # Second phrase features -> similarities
    similar_features = []
    different_features = []
    for index, a_l in enumerate(artwoks_list):
        s, d = watermark_model.compute_features_similarity(a_l, samples_list[index], cal_pix_list[index], cal_latent_list[index])
        if s is not None:
            mean = torch.mean(s, dim=0).squeeze(0)
            mean = mean.expand(s.shape[0], -1, -1, -1)
            s = torch.cat([s,mean], dim=0)
        similar_features.append(s)
        different_features.append(d)
    
    
    # third phrase features -> construct (data, watermark, label)
    train_dataset_arr = []
    for index, a_l in enumerate(artwoks_list):
        dictionary = {'data': a_l,'watermark': artworks_watermarks[index],'label': 1}
        train_dataset_arr.append(dictionary)

        for idx, s_f in enumerate(similar_features[index]):
            for elem in s_f:
                dictionary = {'data': elem,'watermark': artworks_watermarks[index],'label': 1}
                train_dataset_arr.append(dictionary)
        for idx, d_f in enumerate(different_features[index]):
            for elem in d_f:
                # 要保证其watermark不一致
                w_bit= watermark_model.watermark_bit(elem)
                assert w_bit not in artworks_watermarks
                dictionary = {'data': elem,'watermark': w_bit,'label': 0}
                train_dataset_arr.append(dictionary)
    others_set = torch.load(other_path)
    for index, o_s in enumerate(others_set):
        w_bit= watermark_model.watermark_bit(o_s)
        assert w_bit not in artworks_watermarks
        dictionary = {'data': o_s,'watermark': w_bit,'label': 1}
        train_dataset_arr.append(dictionary)
    # forth phrase -> train_loader
    train_dataset = WatermarkDataset(train_dataset_arr)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    lr = opt.lrate
    DDP_multiplier = 1
    print0("Using DDP, lr = %f * %d" % (lr, DDP_multiplier))
    lr *= DDP_multiplier
    optim = get_optimizer([{'params': watermark_decoder.features_watermark.encoder_watermark.parameters(), 'lr': lr * opt.lrate_ratio}], opt, lr=0)
    # 冻结 encoder_features 的参数
    for param in watermark_decoder.features_watermark.encoder_features.parameters():
        param.requires_grad = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_sum = []
    loss_sum_s = []
    loss_sum_l = []
    loss_sum_w = []
    try:
        for ep in range(opt.load_epoch + 1, opt.n_epoch):
            optim.param_groups[1]['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0) # warmup
            optim.param_groups[0]['lr'] = optim.param_groups[1]['lr'] * opt.lrate_ratio
       
            # training
            watermark_decoder.train()
        
            enc_lr = optim.param_groups[0]['lr']
            print(f'epoch {ep}, lr {enc_lr:f}')
            pbar = tqdm(train_loader)
        
            
            # test_data -> 用source target的第一组数据
            test_data_flag = False
            for data, label, watermark in pbar:
                optim.zero_grad()
                source = source.to(device)
                target = target.to(device)
                # 用来test
                if test_data_flag == False:
                    test_data_flag = True
                loss, simclr_loss, label_loss, watermark_loss, simclr_w, best_hyperparameters = watermark_decoder(data, label, watermark)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(parameters=watermark_decoder.features_watermark.encoder_watermark.parameters(), max_norm=opt.grad_clip_norm)
                scaler.step(optim)
                scaler.update()

                # logging
                
                loss = reduce_tensor(loss)
                pbar.set_description(f"loss: {loss:.4f}")
            try:
                loss_sum.append(loss/data.shape[0])
                loss_sum_l.append(simclr_loss/data.shape[0])
                loss_sum_s.append(label_loss/data.shape[0])
                loss_sum_w.append(watermark_loss/data.shape[0])
            except Exception as e:
                print("loss record")
                continue
    except Exception as e:
        dictionary = {
            'loss_sum':loss_sum, 
            'loss_sum_l': loss_sum_l,
            'loss_sum_s': loss_sum_s,
            'loss_sum_w': loss_sum_w,
            'best best_hyperparameters': best_hyperparameters
        }
        import json
        with open('exception.json', "w") as json_file:
            json.dump(dictionary, json_file)
        checkpoint = {
            'MODEL': watermark_decoder.state_dict(),
            'opt': optim.state_dict(),
        }
        loss_vision(loss_sum, loss_sum_s, loss_sum_l, loss_sum_w)
        save_path = os.path.join(model_dir, f"model_{ep}.pth")
        torch.save(checkpoint, save_path)
        print('saved model at', save_path)
    
    checkpoint = {
        'MODEL': watermark_decoder.state_dict(),
        'opt': optim.state_dict(),
    }
    loss_vision(loss_sum, loss_sum_s, loss_sum_l, loss_sum_w)
    save_path = os.path.join(model_dir, f"model_{ep}.pth")
    torch.save(checkpoint, save_path)
    print("best best_hyperparameters:", best_hyperparameters)
    print('saved model at', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/newdata/gluo/ZI/Disentangled_Representation_Learning/soda/config/watermarks.yaml')
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    train(opt)

# /data/LG/AI_Watermarking/Kohya_ss/kohya_ss/wm_version1/art_1_ori_latents.pth
    