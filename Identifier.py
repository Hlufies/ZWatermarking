# Load model directly
import torch.nn as nn
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os

class onwers(nn.Module):
    def __init__(self, device):
        super(onwers, self).__init__()
        self.processor = AutoProcessor.from_pretrained("clip-vit-large-patch14")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("clip-vit-large-patch14").to(device)
    def forward(self, text='swz'):
        inputs = self.processor(text='swz', padding=True, return_tensors="pt")
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        text_features = self.model.get_text_features(**inputs)
        return text_features

device =torch.device('cuda')
onw = onwers(device)
onw.eval()
# self Identifier
o = onw().to(torch.device('cpu'))
# other Identifier
with open('Identifier.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:    
        txt = line.replace('\n', '')
        txt_feature = onw(txt).to(torch.device('cpu'))
        o = torch.cat((o, txt_feature), dim=0)

torch.save(o, 'Identifier.pth')
