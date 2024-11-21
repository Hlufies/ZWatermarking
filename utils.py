import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml
import logging
import json

# ===== Configs =====
def Copy_Model_K(Z_Model, model_k, logger):
    m = 0.999
    try:
        for param_q, param_k in zip(Z_Model.parameters(), model_k.parameters()):
                param_k.data = param_q.data * m
                param_k.requires_grad = False
                param_q.requires_grad = True
    except Exception as e:
        logger.error(f"Copy_Model_K error! :{e}")
        exit(-1)
def Init_First(Z_Model, protectedUnits_list, logger, opt):
    try:
        Z_Model.eval() 
        protectedUnits_wm = []
        for index, protectedUnit in enumerate(protectedUnits_list):
            Z_Model.zero_grad()
            with torch.no_grad():
                w_bit = Z_Model.watermark_bit(protectedUnit.to(opt.device)).to(torch.device('cpu'))
                protectedUnits_wm.append(w_bit)
                torch.cuda.empty_cache()
        del protectedUnit, w_bit  # 释放变量
    except Exception as e:
        logger.error(f"Init First error!: {e}")
        exit(-1)
    return protectedUnits_wm
def Init_Second(Z_Model, protectedUnits_list, Generalization_list, cal_pix_list, cal_latent_list, logger, opt, Valid_NO=False):
    center_features = []
    boundary_features = []
    index_list = []
    opt.test = Valid_NO
    try:
        for index, protectedUnit in enumerate(protectedUnits_list):
            latents_protectedUnit_ = torch.zeros(protectedUnit.shape[0], opt.bit_len,)
            for idx, latent in enumerate(protectedUnit):
                with torch.no_grad():
                    _, t = Z_Model.domainEncoder(latent.unsqueeze(0).to(opt.device))
                    latents_protectedUnit_[idx] = t.to(torch.device('cpu'))

            latents_generalization_ = torch.zeros(Generalization_list[index].shape[0], opt.bit_len)
            for idx, latent in enumerate(Generalization_list[index]):
                with torch.no_grad():
                    _, t = Z_Model.domainEncoder(latent.unsqueeze(0).to(opt.device))
                    latents_generalization_[idx] = t.to(torch.device('cpu'))
            del latent
            del t

            if opt.test:
                index_list.append(index)
                center_features.append(Generalization_list[index].to(torch.device('cpu')))
                boundary_features.append(Generalization_list[index].to(torch.device('cpu')))
            else:
                center_indices, boundary_indices = Z_Model.compute_features_similarity(latents_protectedUnit_, latents_generalization_, cal_pix_list[index], cal_latent_list[index])
                if center_indices.shape[0] >= 1 and boundary_indices.shape[0] >= 1:
                    index_list.append(index)
                else:
                    index_list.append(index)
                    if center_indices.shape[0] <= 0:
                        center_indices = boundary_indices[-1:]
                    elif boundary_indices.shape[0] <=0 :
                        boundary_indices = center_indices[-1:]
                c, d = Generalization_list[index][center_indices].to('cpu'), Generalization_list[index][boundary_indices].to('cpu')
                if c is not None:
                    mean = torch.mean(c, dim=0).squeeze(0)
                    mean = mean.expand(c.shape[0], -1, -1, -1)
                    c = torch.cat([c,mean], dim=0)
                center_features.append(c.to(torch.device('cpu')))
                boundary_features.append(d.to(torch.device('cpu')))
                del c
                del d
            del latents_protectedUnit_
            del latents_generalization_
        del Generalization_list
        del cal_pix_list
        del cal_latent_list
    except Exception as e:
        logger.error(f'Init Second error!: {e}')
        exit(-1)
    '''
        return:
        center_features: 中心样本
        boundary_features: 边缘样本
        index_list: 有效样本路径

    '''
    return center_features, boundary_features, index_list
def Init_Third(opt, protectedUnits_list, protectedUnits_wm, center_features, boundary_features, logger, Valid_NO=False):
    
    train_dataset_array = [] 
    center_samples = []
    try:
        # watermark and z
        '''
        random_streams = [[0] * opt.bit_len] + [[randome_opt.bit] * opt.protectedUnits_num] 
        '''
        random_streams = generate_watermark_label(opt)   
        rare_tokens = torch.load(opt.own_features_path)
        rare_tokens_len = rare_tokens.shape[0]-1  # identifier z 
        opt.test = Valid_NO
        if opt.test:            
            import json
            with open('train_resnet18.json', 'r') as file:
                js = json.load(file)
            js = js[:len(random_streams)]
            for dataindex, data in enumerate(js):
                random_streams[dataindex+1] = data['wm_K']
        
        # for random_stream in random_streams:
        #     print(random_stream[:10])




        '''
            data structure:
                data
                label
                pos_or_neg
                disc
                zws
                wm 

            protected unit -> label > 0, pos_or_neg(center or boundary) -> (1, 0), disc -> 1, wm -> index+1, zws-> 0
            normal unit -> label = 0, pos_or_neg -> 0, disc -> 0, wm -> 全0, zws -> except 0
        '''
        # 正样本

        for index, a_l in enumerate(protectedUnits_list): 
            random_number = 0
            w_bit = protectedUnits_wm[index].squeeze(0)
            dictionary = {'data': a_l.squeeze(0).to(torch.device('cpu')),'label': torch.tensor(index+1).to(torch.device('cpu')), \
                        'pos_or_neg':torch.tensor(1).to(torch.device('cpu')), 'disc':torch.tensor(1).to(torch.device('cpu')), \
                        'zws':rare_tokens[random_number].to(torch.device('cpu')), 'wm':torch.tensor(random_streams[index+1]).to(torch.device('cpu'))}
            # 如果test，就不加入原始样本
            if opt.test==False:
                train_dataset_array.append(dictionary)
                center_samples.append(dictionary)
            for idx, s_f in enumerate(center_features[index]):          
                dictionary = {'data': s_f.to(torch.device('cpu')),'label': torch.tensor(index+1).to(torch.device('cpu')), \
                            'pos_or_neg':torch.tensor(1).to(torch.device('cpu')), 'disc':torch.tensor(1).to(torch.device('cpu')), \
                            'zws':rare_tokens[random_number].to(torch.device('cpu')), 'wm':torch.tensor(random_streams[index+1]).to(torch.device('cpu'))}            
                train_dataset_array.append(dictionary)
            if opt.test:
                # if test, boundary_features = center_features
                # so, continue, ignore!
                continue
            for idx, d_f in enumerate(boundary_features[index]):
                dictionary = {'data': d_f.to(torch.device('cpu')),'label': torch.tensor(index+1).to(torch.device('cpu')), \
                            'pos_or_neg':torch.tensor(0).to(torch.device('cpu')), 'disc':torch.tensor(1).to(torch.device('cpu')),\
                            'zws':rare_tokens[random_number].to(torch.device('cpu')), 'wm':torch.tensor(random_streams[index+1]).to(torch.device('cpu'))} 
                train_dataset_array.append(dictionary)
        # 负样本
        # print(opt.other_path)
        if opt.test==False:
            others_set = torch.load(opt.other_path, map_location=torch.device(opt.device))
            for index, o_s in enumerate(others_set):
                if index >= opt.protectedUnits_nums * opt.every_nums:
                    break
                o_s = o_s.to(torch.device('cpu'))
                assert o_s.device == w_bit.device == torch.device('cpu')
                random_number = random.randint(1, rare_tokens_len)
                dictionary = {'data': o_s.to(torch.device('cpu')),'label': torch.tensor(0).to(torch.device('cpu')), \
                            'pos_or_neg':torch.tensor(0).to(torch.device('cpu')), 'disc':torch.tensor(0).to(torch.device('cpu')),\
                            'zws':rare_tokens[random_number].to(torch.device('cpu')), 'wm':torch.tensor(random_streams[0]).to(torch.device('cpu'))}
                train_dataset_array.append(dictionary)
            del others_set
    except Exception as e:
        logger.error(f'Init Third error!: {e}')
        exit(-1)
    return train_dataset_array, random_streams, center_samples
def save_watermark(opt, protectedUnits_path, random_streams, logger):
    try:
        watermark_json = []
        for i in range(len(protectedUnits_path)):
            watermark_json.append({
                'domain_center':protectedUnits_path[i],
                'wm_K':random_streams[i+1],
                })
        with open(opt.save_name+'.json', 'w') as f:
            json.dump(watermark_json, f, indent=4)
    except Exception as e:
        logger.error(f"save watermark error! : {e}")
        exit(0)
def load_state_dict(Model, pretrained_model, device):
    if pretrained_model is not None:
        Model.load_state_dict(pretrained_model)
    return Model
def load_dir(opt, current_script_directory:str=None, logger:logging.Logger=None)->list:
    # The absspath of utils.py 
    if current_script_directory is None:
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
    tsbd_dir = os.path.join(opt.save_dir, "tensorboard")
    log_dir = os.path.join(current_script_directory, 'logging')
    check_dir_exists(tsbd_dir)
    check_dir_exists(log_dir)
    return [tsbd_dir, log_dir]
def load_pretrained(opt, current_script_directory:str=None, logger:logging.Logger=None)->list:
    
    # The absspath of utils.py 
    if current_script_directory is None:
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(opt.save_dir, "ckpts")
    model_path = current_script_directory+model_dir.split('.')[-1]
    pretrained_model_path = os.path.join(model_path, f'{opt.protectedUnits_nums}_encoder.pth')
    pretrained_optim_path = os.path.join(model_path, 'optim.pth')
    state1 = check_file_exitst(pretrained_model_path, logger, 'pretrained_model', False)
    state2 = check_file_exitst(pretrained_optim_path, logger, "pretrained_optim", False)
    pretrained_model, pretrained_optim = None, None
    if state1:
        pretrained_model = torch.load(pretrained_model_path, map_location='cpu')
    if state2:
        pretrained_optim = torch.load(pretrained_optim_path, map_location='cpu')
    return [pretrained_model, pretrained_optim]

def log(log_dir:str=None) -> logging.Logger:
    log_file = os.path.join(log_dir, 'app.log')  # 日志文件路径
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[92m',  # 绿色
            'INFO': '\033[94m',   # 蓝色
            'WARNING': '\033[91m', # 黄色（默认警告）
            'ERROR': '\033[91m',  # 红色
        }

        RESET = '\033[0m'  # 重置颜色
        TIME_COLOR = '\033[1;32m'  # 绿色加粗
        def format(self, record):
            # 格式化时间
            original_asctime = self.formatTime(record, self.datefmt)
            record.asctime = f"{self.TIME_COLOR}{original_asctime}{self.RESET}"
            color = self.COLORS.get(record.levelname, self.RESET)
            record.msg = f"{color}{record.msg}{self.RESET}"
            return super().format(record)

    # 创建一个日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # 创建一个文件处理器并设置其格式
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s:%(message)s', datefmt='%Y-%m-%d %H:%M')
    file_handler.setFormatter(file_formatter)
    # 创建一个控制台处理器并设置其格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M'))
    # 添加处理器到日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def check_file_exitst(path:str, logger:logging.Logger, info:str=None, info_show:bool=True) -> bool:
    if os.path.exists(path) == False:
        logger.error(f'{path} is not found!')
        return False
    else:
        if info is not None:
            if info_show:
                logger.info(f'{info}: {path}')
        else:
            if info_show:
                logger.info(f'{path}')
        return True
        
def check_dir_exists(path:str):
    # if don't exist, make dir 
    if os.path.exists(path) == False:
        os.makedirs(path, exist_ok=True)

class Config(object):
    def __init__(self, opt):
        yaml_path = opt.config
        use_amp = opt.use_amp
        with open(yaml_path, 'r') as f:
            opt = yaml.full_load(f)
        for key in opt:
            setattr(self, key, opt[key])

def get_optimizer(parameters, opt, lr):
    if not hasattr(opt, 'optim'):
        return torch.optim.Adam(parameters, lr=lr)
    elif opt.optim == 'AdamW':
        return torch.optim.AdamW(parameters, **opt.optim_args, lr=lr)
    else:
        raise NotImplementedError()
def init_optimizer(optim, optim_params):
    for key, value in optim_params.items():
        if key == 'params':
            continue
        optim.param_groups[0][key] = value

    return optim
# ===== Multi-GPU training =====

def init_seeds(RANDOM_SEED=1337, no=0):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def gather_tensor(tensor):
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list


def DataLoaderDDP(dataset, batch_size, shuffle=True):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        # pin_memory=True,
    )
    return dataloader, sampler

def print0(*args, **kwargs):
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        print(*args, **kwargs)

def adaptive_bit_len(opt, random_streams):
    if opt.bit_len < 128:
        random_streams = [random_streams[i][0:opt.bit_len] for i in range(len(random_streams))]
    elif opt.bit_len > 128:
        random_streams_ = []
        r = int(opt.bit_len / 128)
        l = len(random_streams)
        for i in range(l):
            tmp = random_streams[i].copy()
            for j in range(r-1):
                random_streams[i] += tmp
    return random_streams

def generate_watermark_label(opt):
    stream_length = opt.bit_len
    num_streams = opt.protectedUnits_nums  # 要保护的unit数目
    random_streams = []
    random_streams_dict = {} # 用字典检查水印是否重复

    # 负样本水印域 全0
    random_stream = [random.randint(0, 0) for _ in range(stream_length)]
    random_streams.append(random_stream)
    tmp = ""
    for s_l in random_stream:
        tmp+=str(s_l)          
    random_streams_dict[tmp] = 1  

    # 正样本水印域 全0
    while num_streams:
        tmp=""
        random_stream = [random.randint(0, 1) for _ in range(stream_length)]
        for s_l in random_stream:
            tmp+=str(s_l)       
        if tmp in random_streams_dict.keys():
            continue
        else:
            random_streams_dict[tmp]=1
        random_streams.append(random_stream)
        num_streams -= 1
    random_streams = adaptive_bit_len(opt, random_streams)
    return random_streams
