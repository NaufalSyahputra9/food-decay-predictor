# api/db/models.py — SQLAlchemy + SQLite
from sqlalchemy import Column, String, Integer, Float, Boolean, LargeBinary, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase
import uuid
import datetime

class Base(DeclarativeBase):
    pass
\

class FoodSession(Base):
    __tablename__ = "food_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    food_label = Column(String(100), nullable=False)
    is_known = Column(Boolean, default=True)  # Dikenal OSR atau via CLIP
    storage_type = Column(String(20), default="room_temp")
    start_date = Column(String, default=lambda: datetime.date.today().isoformat())
    is_active = Column(Boolean, default=True)

class DailySnapshot(Base):
    __tablename__ = "daily_snapshots"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False) 
    day_index = Column(Integer, nullable=False) 
    photo_path = Column(String)
    captured_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

# Buat tabel yang isinya hasil ekstraksi fitur (vektor pakai CLIP) dari gambar
class FeatureCache(Base):
    __tablename__ = "feature_cache"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_id = Column(String, nullable=False)  
    feature_data = Column(LargeBinary)            
    feature_type = Column(String(10), default="cnn") # "cnn" atau "clip"
    shape_info = Column(JSON)                     

# Buat tabel hasil prediksi
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)   # Relasi manual ke FoodSession
    days_remaining = Column(Float)
    freshness_status = Column(String(20))         # fresh / caution / warning / expired
    confidence = Column(Float)
    decay_curve = Column(JSON)                    # Contoh: [3.0, 1.8, 0.4, ...]
    predicted_at = Column(String, default=lambda: datetime.datetime.now().isoformat())