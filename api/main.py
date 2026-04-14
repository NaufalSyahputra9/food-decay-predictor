# api/main.py
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
import os
import shutil
import numpy as np
import torch
import uuid 
from fastapi.staticfiles import StaticFiles

from api.db.session import get_db
from api.db import crud, models
from api import schemas
from models.pipeline import extract_features, predict_sequence, _get_models

app = FastAPI(title="ChronoFood API", version="1.0")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_get_models(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

@app.post("/predict", response_model=schemas.PredictionResult)
async def predict_shelflife(
    file: UploadFile = File(...),
    session_id: str = Form(None, description="Kosongkan jika hari pertama"),
    storage_type: str = Form("room_temp"),
    db: Session = Depends(get_db)
):
    # 1. Simpan foto (hanya sekali)
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    permanent_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(permanent_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Ekstraksi fitur dari file
    hasil_ekstraksi = extract_features(permanent_path)
    
    is_known = hasil_ekstraksi["is_known"]
    ai_label = hasil_ekstraksi["food_label"]
    confidence = hasil_ekstraksi["confidence"]
    
    # Jika benda asing, langsung return hasil tanpa masukin ke DB
    if not is_known:
        try:
            if os.path.exists(permanent_path):
                os.remove(permanent_path)
        except Exception:
            pass #
            
        return schemas.PredictionResult(
            session_id=session_id or "rejected",
            days_remaining=-1.0,
            freshness_status="unknown",
            confidence=confidence,
            decay_curve=None,
            food_label=ai_label
        )

    # Jika makanan dikenal, lanjutkan ke langkah penyimpanan dan prediksi ConvLSTM
    fitur_tensor = hasil_ekstraksi["feature_tensor"] # [C, H, W]
    feature_bytes = fitur_tensor.numpy().tobytes()
    shape_info = {"C": fitur_tensor.shape[0], "H": fitur_tensor.shape[1], "W": fitur_tensor.shape[2]}

    is_first_day = session_id is None
    
    if is_first_day:
        session_data = schemas.FoodSessionCreate(food_label=ai_label, storage_type=storage_type)
        db_session = crud.create_session(db, session_data, is_known=True)
        current_session_id = db_session.id
        day_index = 0
    else:
        db_session = crud.get_food_session(db, session_id)
        if not db_session:
            # Cegah nyampah kalau session-nya salah
            if os.path.exists(permanent_path):
                os.remove(permanent_path)
            raise HTTPException(status_code=404, detail="Session ID tidak ditemukan")
            
        if ai_label != db_session.food_label:
            if os.path.exists(permanent_path):
                os.remove(permanent_path)
                
            return schemas.PredictionResult(
                session_id=session_id,
                is_known_food=True,
                days_remaining=-1.0,
                freshness_status="error_mismatch",
                confidence=confidence,
                decay_curve=None,
                food_label=ai_label
            )
            
        current_session_id = session_id
        
        # Hitung ini hari ke berapa
        past_snapshots = db.query(models.DailySnapshot).filter_by(session_id=current_session_id).all()
        day_index = len(past_snapshots)

    # 3. Simpan metadata snapshot & fitur ke DB
    snap_data = schemas.SnapshotUpload(session_id=current_session_id, day_index=day_index)
    # Masukkan permanent_path (bukan kata "deleted") agar UI bisa melacak gambarnya
    db_snap = crud.save_snapshot(db, snap_data, photo_path=permanent_path) 
    crud.save_feature(db, db_snap.id, feature_bytes, "cnn", shape_info)

    # 4. Tarik semua fitur yang ada untuk sesi ini, lalu prediksi dengan ConvLSTM
    sequence_records = crud.get_sequence(db, current_session_id)
    T = len(sequence_records)
    WINDOW_SIZE = 3 

    if T < WINDOW_SIZE:
        # Belum cukup hari
        return schemas.PredictionResult(
            session_id=current_session_id,
            days_remaining=-1.0,
            freshness_status="collecting_data",
            confidence=confidence,
            decay_curve=None
        )
    
    tensors_list = []
    for record in sequence_records:
        s = record.shape_info
        arr = np.frombuffer(record.feature_data, dtype=np.float32).reshape((s["C"], s["H"], s["W"]))
        tensors_list.append(torch.from_numpy(arr))
        
    # Masuk ke pipeline model 3 
    hasil_prediksi = predict_sequence(tensors_list)
    
    crud.save_prediction(
        db, current_session_id, 
        days=hasil_prediksi["days_remaining"], 
        status=hasil_prediksi["status"], 
        confidence=0.88
    )

    return schemas.PredictionResult(
        session_id=current_session_id,
        days_remaining=hasil_prediksi["days_remaining"],
        freshness_status=hasil_prediksi["status"],
        confidence=0.88,
        decay_curve=None
    )


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/sessions") 
async def get_all_sessions(db: Session = Depends(get_db)):
    """Mengambil semua sesi yang pernah dibuat di database."""
    sessions = db.query(models.FoodSession).order_by(models.FoodSession.start_date.desc()).all()
    
    data_aman = []
    for s in sessions:
        data_aman.append({
            "id": s.id,
            "food_label": s.food_label,
            "start_date": s.start_date
        })
        
    return data_aman

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Mengambil riwayat foto dan status sebuah sesi untuk ditampilkan di UI."""
    db_session = crud.get_food_session(db, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Sesi tidak ditemukan")
        
    snapshots = db.query(models.DailySnapshot).filter_by(session_id=session_id).order_by(models.DailySnapshot.day_index).all()
    
    history_data = []
    for snap in snapshots:
        url_path = snap.photo_path.replace("data/uploads", "/uploads").replace("\\", "/")
        history_data.append({
            "day": snap.day_index + 1, #
            "image_url": f"http://127.0.0.1:8000{url_path}",
            "captured_at": snap.captured_at
        })
        
    return {
        "food_label": db_session.food_label,
        "is_known": db_session.is_known,
        "total_days": len(history_data),
        "history": history_data
    }