# api/main.py
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
import os
import shutil
from PIL import Image # Tambahan untuk baca gambar

# Import struktur lokal kita
from api.db.session import get_db
from api.db import crud, models
from api import schemas

# IMPORT SANG MANAJER AI
from models.pipeline import ShelfLifePipeline

app = FastAPI(title="ChronoFood API", version="1.0")

# Pastikan folder penyimpanan foto harian tersedia
os.makedirs("data/raw/temporal", exist_ok=True)

# ==========================================================
# INISIALISASI PIPELINE AI (Jalan 1x saat server start)
# ==========================================================
print("Menyiapkan AI Pipeline...")
ai_pipeline = ShelfLifePipeline()

@app.post("/predict", response_model=schemas.PredictionResult)
async def predict_shelflife(
    file: UploadFile = File(...),
    session_id: str = Form(None, description="Kosongkan jika ini hari pertama"),
    food_label: str = Form("unknown", description="Label makanan awal"),
    storage_type: str = Form("room_temp", description="room_temp / fridge / freezer"),
    db: Session = Depends(get_db)
):
    """
    Endpoint utama untuk Inference Pipeline (P1 - P6)
    """
    # ---------------------------------------------------------
    # P1: User Upload Foto + Metadata
    # ---------------------------------------------------------
    file_path = f"data/raw/temporal/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Buka gambar pakai PIL untuk dimasukkan ke AI
    image = Image.open(file_path).convert("RGB")
        
    # ---------------------------------------------------------
    # P2 & P3: Ekstraksi Fitur via AI (Open-Set & CLIP)
    # ---------------------------------------------------------
    # Memanggil AI sungguhan menggantikan simulasi!
    fitur_tensor, ai_label, is_known = ai_pipeline.proses_gambar_baru(image)
    
    # Ubah tensor PyTorch menjadi bytes agar bisa disimpan ke SQLite
    feature_bytes = fitur_tensor.cpu().numpy().tobytes()
    shape_info = {"C": fitur_tensor.shape[1], "H": fitur_tensor.shape[2], "W": fitur_tensor.shape[3]}

    is_first_day = session_id is None
    
    if is_first_day:
        session_data = schemas.FoodSessionCreate(food_label=ai_label, storage_type=storage_type)
        db_session = crud.create_session(db, session_data, is_known=is_known)
        current_session_id = db_session.id
        day_index = 0
    else:
        db_session = crud.get_food_session(db, session_id)
        if not db_session:
            raise HTTPException(status_code=404, detail="Session ID tidak ditemukan")
        
        current_session_id = session_id
        past_snapshots = crud.get_snapshots_by_session(db, current_session_id)
        day_index = len(past_snapshots)
        ai_label = db_session.food_label # Tetap gunakan label dari hari pertama
        is_known = db_session.is_known

    # Simpan metadata foto hari ini
    snapshot_data = schemas.SnapshotUpload(session_id=current_session_id, day_index=day_index)
    db_snapshot = crud.save_snapshot(db, snapshot_data, photo_path=file_path)

    # Simpan vektor hasil ekstraksi AI ke DB
    crud.save_feature(db, db_snapshot.id, feature_bytes, "cnn" if is_known else "clip", shape_info)

    # ---------------------------------------------------------
    # P4: Sequence Assembly dari Database
    # ---------------------------------------------------------
    sequence_data = crud.get_sequence(db, current_session_id)
    T = len(sequence_data) 

    # ---------------------------------------------------------
    # P5: ConvLSTM Inference
    # ---------------------------------------------------------
    if T < 3:
        pred_days = 7.0
        status = "fresh"
        confidence = 0.50
        curve = None
        recommendation = "Data masih kurang. Lanjutkan foto besok untuk prediksi ConvLSTM."
    else:
        # Memanggil ConvLSTM sungguhan!
        pred_days = ai_pipeline.prediksi_sisa_hari(sequence_data)
        
        # Logika status sederhana berdasarkan sisa hari
        if pred_days > 5: status = "fresh"
        elif pred_days > 2: status = "caution"
        else: status = "warning"
        
        confidence = 0.88
        curve = [] # Decay curve bisa diisi dari iterasi prediksi nanti
        recommendation = f"Segera konsumsi dalam {int(pred_days)} hari ke depan."

    # ---------------------------------------------------------
    # P6: Post-processing & Simpan Prediksi
    # ---------------------------------------------------------
    crud.save_prediction(db, current_session_id, pred_days, status, confidence, curve)

    return schemas.PredictionResult(
        session_id=current_session_id,
        food_label=ai_label,
        is_known_food=is_known,
        days_remaining=pred_days,
        freshness_status=status,
        confidence=confidence,
        decay_curve=curve,
        recommendation=recommendation
    )