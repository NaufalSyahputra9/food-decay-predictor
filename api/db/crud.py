# api/db/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import asc
import uuid

from api.db import models
from api import schemas

def create_session(db: Session, session_data: schemas.FoodSessionCreate, is_known: bool = True):
    """Membuat sesi makanan baru (Langkah P2/P3)."""
    db_session = models.FoodSession(
        id=str(uuid.uuid4()),
        food_label=session_data.food_label,
        is_known=is_known,
        storage_type=session_data.storage_type
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def save_snapshot(db: Session, snapshot_data: schemas.SnapshotUpload, photo_path: str):
    """Menyimpan metadata foto harian (Langkah P1)."""
    db_snapshot = models.DailySnapshot(
        id=str(uuid.uuid4()),
        session_id=snapshot_data.session_id,
        day_index=snapshot_data.day_index,
        photo_path=photo_path
    )
    db.add(db_snapshot)
    db.commit()
    db.refresh(db_snapshot)
    return db_snapshot

def save_feature(db: Session, snapshot_id: str, feature_data: bytes, feature_type: str, shape_info: dict):
    """Menyimpan feature map (CNN/CLIP) dalam bentuk bytes (Langkah P6)."""
    db_feature = models.FeatureCache(
        id=str(uuid.uuid4()),
        snapshot_id=snapshot_id,
        feature_data=feature_data,
        feature_type=feature_type,
        shape_info=shape_info
    )
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature

def get_sequence(db: Session, session_id: str):
    """
    Menarik riwayat feature map secara berurutan untuk ConvLSTM (Langkah P4).
    Menggabungkan tabel FeatureCache dan DailySnapshot berdasarkan snapshot_id.
    """
    results = db.query(models.FeatureCache)\
                .join(models.DailySnapshot, models.FeatureCache.snapshot_id == models.DailySnapshot.id)\
                .filter(models.DailySnapshot.session_id == session_id)\
                .order_by(asc(models.DailySnapshot.day_index))\
                .all()
    return results

def save_prediction(db: Session, session_id: str, days: float, status: str, confidence: float, curve: list = None):
    """Menyimpan hasil akhir prediksi ConvLSTM ke database (Langkah P6)."""
    db_prediction = models.Prediction(
        id=str(uuid.uuid4()),
        session_id=session_id,
        days_remaining=days,
        freshness_status=status,
        confidence=confidence,
        decay_curve=curve
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction