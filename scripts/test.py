import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import torch.nn.functional as F
import os
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# Memanggil arsitektur AI dari folder models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from models.open_set import OpenSetClassifier


# =====================================================================
# 3. ARSITEKTUR MODEL 1 (OPEN-SET CLASSIFIER dgn ENERGY-BASED)
# =====================================================================
class OpenSetClassifier(nn.Module):
    def __init__(self, n_classes, energy_threshold=-10.0):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # Freeze sebagian besar layer agar cepat
        for name, p in self.backbone.named_parameters():
            if not any(name.startswith(s) for s in ['features.6', 'features.7', 'classifier']):
                p.requires_grad = False

        # Ganti kepala classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1536, n_classes)
        )
        self.energy_threshold = energy_threshold

    def forward(self, x):
        feat_map = self.backbone.features[:6](x)
        out = self.backbone.features[6:](feat_map)
        out = self.backbone.avgpool(out)
        logits = self.backbone.classifier(out.flatten(1))

        energy = -torch.logsumexp(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        return {
            "logits": logits,
            "pred_class": pred,
            "confidence": conf,
            "energy": energy,
            "is_known": (energy < self.energy_threshold),
            "feature_map": feat_map
        }



def test_gambar_saya(image_path):
    print(f"🔍 Memproses gambar: {image_path}...")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Kamus Buah (Sesuaikan urutan abjad foldermu di Colab kemarin)
    # Kalau urutan foldernya alpukat, pir, pisang, tomat:
    daftar_buah = ["alpukat", "pir", "pisang", "tomat"]
    
    # 3. Load Model & Weights
    # Kita pakai threshold -10.0 dulu (bisa diubah nanti)
    model = OpenSetClassifier(n_classes=len(daftar_buah), energy_threshold=-10.0)
    
    try:
        model.load_state_dict(torch.load("weights/opensetv2.pt", map_location=device))
        print("✅ Weights berhasil dimuat!")
    except Exception as e:
        print("❌ Gagal memuat weights. Pastikan file 'weights/openset.pt' ada!")
        return
    
    model.to(device)
    model.eval() # PENTING: Kunci mode agar tidak belajar lagi
    
    # 4. Transformasi Gambar (SAMA PERSIS DENGAN SAAT VALIDATION DI COLAB)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 5. Baca dan Proses Gambar
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device) # Tambah dimensi batch (1, C, H, W)
    except Exception as e:
        print(f"❌ Gagal membaca gambar: {e}")
        return
    
    # 6. Prediksi!
    with torch.no_grad():
        hasil = model(image_tensor)
        # ── DIAGNOSTIC BLOCK ──────────────────────────────────────
        print("\n📊 DIAGNOSTIC INFO:")
        print(f"Raw logits     : {hasil['logits'].detach().cpu().numpy()}")
        print(f"Logit min/max  : {hasil['logits'].min().item():.3f} / {hasil['logits'].max().item():.3f}")
        print(f"Logit std dev  : {hasil['logits'].std().item():.4f}")  # kalau < 1.0, model belum terlatih
        print(f"Energy score   : {hasil['energy'].item():.4f}")
        print(f"Theoretical random baseline (4 kelas): {-torch.log(torch.tensor(4.0)).item():.4f}")

        # Cek apakah logits mendekati uniform (tanda model belum terlatih)
        logit_range = hasil['logits'].max().item() - hasil['logits'].min().item()
        if logit_range < 2.0:
            print("\nPERINGATAN: Logit range terlalu kecil ({:.3f}).".format(logit_range))
            print("   Ini tanda model BELUM TERLATIH atau UNDERFIT.")
            print("   Selesaikan training notebook 01 dulu sebelum testing di sini.")
        else:
            print(f"\n✅ Logit range OK ({logit_range:.3f}) — model sudah mulai belajar.")

        
    angka_tebakan = hasil["pred_class"].item()
    nama_tebakan = daftar_buah[angka_tebakan]
    persentase = hasil["confidence"].item() * 100
    skor_energi = hasil["energy"].item()
    dikenali = hasil["is_known"].item()
    
    # 7. Tampilkan Hasil
    print("\n" + "="*40)
    print("🎯 HASIL PREDIKSI AI:")
    print("="*40)
    if dikenali:
        print(f"✅ Makanan Dikenali : YA (Lanjut ke ConvLSTM)")
        print(f"🍎 Tebakan Buah   : {nama_tebakan.upper()}")
        print(f"📊 Tingkat Yakin    : {persentase:.2f}%")
    else:
        print(f"🚨 Makanan Dikenali : TIDAK (Benda Asing! Lempar ke CLIP)")
        print(f"❓ Tebakan Asal     : {nama_tebakan.upper()} (Diabaikan karena ragu)")
    
    print(f"⚡ Skor Energi      : {skor_energi:.2f} (Batas threshold: {model.energy_threshold})")
    print("="*40 + "\n")

if __name__ == "__main__":
    abs_path = r"C:\Users\Naufal Syahputra\Documents\KCVanguard_When-yah\FP-2026\food-decay-predictor\scripts"
    #files = ["test_avocado.jpeg", "test_banana.jpg", "test_pir.jpg", "test_tomato.jpg"]
    files = ["cat.jpg"]
    for f in files:
        FOTO_TEST = os.path.join(abs_path, f)  
        if os.path.exists(FOTO_TEST):
            test_gambar_saya(FOTO_TEST)
        else:
            print(f"Taruh dulu file {FOTO_TEST} di folder project ya!")
        
        print("\n")