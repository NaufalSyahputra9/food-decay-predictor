import clip
import torch
import torch.nn as nn
import pathlib 
import os 
import numpy
import matplotlib


WEIGHTS_DIR = os.path("../weights")
DATA_ROOT = os.path("../data/raw/food101")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

for param in clip_model.parameters():
  param.requires_grad = False

class CLIPAdapter(nn.Module):
    #
    def __init__(self, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.adapter = nn.Linear(512, num_classes)

    # 
    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
        return self.adapter(features)
    
