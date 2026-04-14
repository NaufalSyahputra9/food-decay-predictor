import clip
import torch
import torch.nn.functional as F
from PIL import Image

class ZeroShotIdentifier:
    """
    Dipanggil pipeline HANYA saat OSR mendeteksi is_known=False (Benda Asing).
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def identify(self, pil_image, candidates: list[str]):
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensor)          
            img_feat = F.normalize(img_feat, dim=-1)

        templates = [
            "a photo of {}",
            "a photo of fresh {}",
            "a close up photo of {}",
        ]
        text_feats = []
        for template in templates:
            texts = clip.tokenize(
                [template.format(c) for c in candidates]
            ).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_text(texts)    
                feat = F.normalize(feat, dim=-1)
            text_feats.append(feat)

        text_feat_avg = torch.stack(text_feats).mean(dim=0)         
        text_feat_avg = F.normalize(text_feat_avg, dim=-1)
        logit_scale = self.model.logit_scale.exp()

        sims = (logit_scale *img_feat @ text_feat_avg.T).squeeze(0)              
        sims = sims.softmax(dim=-1)                                 

        best_idx = sims.argmax().item()
        best_label = candidates[best_idx]
        confidence = sims[best_idx].item()

        return best_label, confidence


