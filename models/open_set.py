# Arsitektur Open Set Classifier berbasis EfficientNet-B3.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class OpenSetClassifier(nn.Module):
    def __init__(self, n_classes, energy_threshold=-10.0):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        for name, p in self.backbone.named_parameters():
            if not any(name.startswith(s) for s in
                       ['features.4', 'features.5', 'features.6', 'features.7', 'classifier']):
                p.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1536, n_classes)
        )
        self.energy_threshold = energy_threshold

    def forward(self, x):
        feat_map = self.backbone.features[:6](x)
        out      = self.backbone.features[6:](feat_map)
        out      = self.backbone.avgpool(out)
        flat_out = out.flatten(1) 
        
        logits   = self.backbone.classifier(flat_out)
        energy   = -torch.logsumexp(logits, dim=-1)
        probs    = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        
        return {
            "logits": logits, 
            "pred_class": pred,
            "confidence": conf, 
            "energy": energy,
            "is_known": (energy < self.energy_threshold),
            "feature_map": feat_map,
            "feature_64d": flat_out[:, :64] 
        }