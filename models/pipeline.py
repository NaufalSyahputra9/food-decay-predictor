# Arsitektur pipeline utama untuk ekstraksi fitur dan prediksi sisa usia simpan makanan.
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import sys, os, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.open_set   import OpenSetClassifier
from models.clip_adapter import ZeroShotIdentifier
from models.convlstm_bone   import ShelfLifePredictor

def _load_models(device: torch.device):
    osr_cfg_path = os.path.join(BASE_DIR, "weights", "osr_config.json")
    with open(osr_cfg_path) as f: osr_cfg = json.load(f)

    clstm_cfg_path = os.path.join(BASE_DIR, "weights", "convlstm_config.json")
    with open(clstm_cfg_path) as f: clstm_cfg = json.load(f)

    candidates_path = os.path.join(BASE_DIR, "models", "candidates.json")
    with open(candidates_path) as f: kandidat_asing = json.load(f)["kandidat_asing"]

    osr_weights = os.path.join(BASE_DIR, "weights", "opensetv5.pt")
    osr_model   = OpenSetClassifier(n_classes=osr_cfg["n_classes"], energy_threshold=osr_cfg["energy_threshold"]).to(device)
    osr_model.load_state_dict(torch.load(osr_weights, map_location=device))
    osr_model.eval()

    extractor = osr_model.backbone.features[:6].to(device).eval()

    clip_model = ZeroShotIdentifier(device=device)

    clstm_weights = os.path.join(BASE_DIR, "weights", "convlstm.pt")
    clstm_model = ShelfLifePredictor(
        in_channels=clstm_cfg["in_channels"],
        hidden1=clstm_cfg.get("hidden1", 16),
        hidden2=clstm_cfg.get("hidden2", 8)
    ).to(device)
    clstm_model.load_state_dict(torch.load(clstm_weights, map_location=device))
    clstm_model.eval()

    return {
        "osr": osr_model, "extractor": extractor, "clip": clip_model, "convlstm": clstm_model,
        "osr_cfg": osr_cfg, "clstm_cfg": clstm_cfg, "kandidat": kandidat_asing,
    }

_MODEL_CACHE = None
def _get_models(device):
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_models(device)
    return _MODEL_CACHE

def _get_status(days: float) -> str:
    if days > 5:  return "fresh"
    if days > 2:  return "caution"
    return "warning"

def _get_recommendation(days: float) -> str:
    if days > 5:  return f"Masih segar, sisa sekitar {days:.1f} hari."
    if days > 2:  return f"Segera konsumsi dalam {days:.1f} hari ke depan."
    return "Konsumsi hari ini, batas kelayakan hampir habis."

def extract_features(image_path: str) -> dict:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models  = _get_models(device)
    osr_cfg = models["osr_cfg"]
    class_names = osr_cfg["class_names"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        osr_out  = models["osr"](img_t)

    is_known   = bool(osr_out["is_known"].item())
    pred_class = int(osr_out["pred_class"].item())
    confidence = float(osr_out["confidence"].item())
    
    if not is_known:
        nama_asing, conf_clip = models["clip"].identify(image, models["kandidat"])
        return {
            "is_known": False,
            "food_label": nama_asing,
            "confidence": conf_clip,
            "feature_tensor": None
        }

    with torch.no_grad():
        feat_map = models["extractor"](img_t).cpu().squeeze(0)  # [C, H, W]

    return {
        "is_known": True,
        "food_label": class_names[pred_class],
        "confidence": confidence,
        "feature_tensor": feat_map
    }

def predict_sequence(tensors_list: list) -> dict:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models  = _get_models(device)
    
    # Ambil window_size frame terakhir
    window_size = models["clstm_cfg"]["window_size"]
    window_feats = tensors_list[-window_size:] 
    
    seq_tensor = torch.stack(window_feats).unsqueeze(0).to(device)  # [1, T, C, H, W]

    with torch.no_grad():
        pred_raw = models["convlstm"](seq_tensor).item()

    days_remaining = max(0.0, pred_raw)
    return {
        "days_remaining": days_remaining,
        "status": _get_status(days_remaining),
        "recommendation": _get_recommendation(days_remaining)
    }
