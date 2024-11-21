from torch.cuda.amp import autocast as autocast
import torch


def metrics_disc(disc, disc_indices, metrics_dicriminator_disc):
    samples_indices = torch.nonzero(disc.int()).squeeze(-1)
    nosise_indices =  torch.nonzero(disc.int() == 0).squeeze()
    if samples_indices.numel() != 0:
        metrics_dicriminator_disc[0] += torch.sum(disc[samples_indices].int() == disc_indices[samples_indices].int())
        metrics_dicriminator_disc[2] += torch.sum(disc[samples_indices].int() != disc_indices[samples_indices].int())
    if nosise_indices.numel() != 0:
        metrics_dicriminator_disc[1] += torch.sum(disc[nosise_indices].int() == disc_indices[nosise_indices].int())
        metrics_dicriminator_disc[3] += torch.sum(disc[nosise_indices].int() != disc_indices[nosise_indices].int())
    return metrics_dicriminator_disc
def metrics_class(disc,class_labels_clone, wm_classes_logits, metrics_dicriminator_cls):
    samples_indices = torch.nonzero(disc.int()).squeeze(-1)
    nosise_indices =  torch.nonzero(disc.int() == 0).squeeze()
    max_labels_indices = torch.argmax(class_labels_clone , dim=1) 
    max_indices = torch.argmax(wm_classes_logits, dim=1)   
    if samples_indices.numel() != 0:
        metrics_dicriminator_cls[0] += torch.sum(max_labels_indices[samples_indices].int() == max_indices[samples_indices].int()) 
        metrics_dicriminator_cls[2] += torch.sum(max_labels_indices[samples_indices].int() != max_indices[samples_indices].int())
    if nosise_indices.numel() != 0:
        metrics_dicriminator_cls[1] += torch.sum(max_labels_indices[nosise_indices].int() == max_indices[nosise_indices].int())
        metrics_dicriminator_cls[3] += torch.sum(max_labels_indices[nosise_indices].int() != max_indices[nosise_indices].int())
    return metrics_dicriminator_cls  
def metrics_wm(disc, binary_watermark_int, wm, dict_w_d = {}):
    
    samples_indices = torch.nonzero(disc.int()).squeeze(-1)
    samples_nums = 0
    if samples_indices.numel() != 0:
        pre_wm = binary_watermark_int[samples_indices] # prediction
        ori_wm = wm[samples_indices]                   # wm
        for b_index, (bwls, bws) in enumerate(zip(ori_wm, pre_wm)):      
            key = torch.sum(bwls.int() == bws.int()).item()
            if key not in dict_w_d.keys():
                dict_w_d[key] =1
            else:
                dict_w_d[key] += 1
            samples_nums += 1       
    return dict_w_d, samples_nums 

def cal_Confusion_Matrix(metrics_dicriminator):
    '''
    metrics_dicriminator = [0, 0, 0, 0] # TP, TN, FP, FN 
    TP: True Positive
    TN: True Negative
    FP: False Positive
    FN: False Negative
    '''
    [TP, TN, FP, FN] =metrics_dicriminator
    # precesion
    precesion = TP/(TP+FP)
    # accuracy
    accuracy = TP/(TP+TN+FP+FN)
    # recall
    recall = TP/(TP+TN)
    # F1
    F1 = 2*recall*precesion/(recall+precesion)
    return [precesion, accuracy, recall, F1]

def cal_rate(opt, dict_w_d, num_elements):
    
    rate = [0 for i in range(10)]
    rate_thd = [0.1*(i+1) for i in range(10)]
    
    best_nums = 0
    avg_acc = 0

    sums = 0
    print(dict_w_d)
    for k, v in dict_w_d.items():
        sums += v
    assert sums == num_elements

    for k, v in dict_w_d.items():    
        bit_acc = k/opt.bit_len
        rate_idex = None
        for idex, thd in enumerate(rate_thd):
            if bit_acc <= thd:
                rate_idex = idex
                break
        assert rate_idex is not None
        rate[rate_idex] += v
        avg_acc += v * k
        if k == opt.bit_len:
            best_nums = v 
    valid_sums = 0
    for v in rate:
        valid_sums += v
    assert valid_sums == sums and valid_sums == num_elements
    avg_acc = (avg_acc/128)/num_elements
    best_rate = best_nums / num_elements
    str_out = ''
    for rt, t in zip(rate_thd, rate):
        str_out += f'|rate {rt:.1f}:{(t/num_elements):.4f}'
    return rate, rate_thd, best_rate, avg_acc, str_out
