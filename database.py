import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Track(Base):
    """Track metadata."""
    __tablename__ = "tracks"

    id = Column(Text, primary_key=True)
    name = Column(Text)
    duration_ms = Column(Integer)
    popularity = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class User(Base):
    """User account information."""
    __tablename__ = "users"

    id = Column(Text, primary_key=True)
    email = Column(Text)
    name = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class SongFeature(Base):
    """Audio features for content-based filtering."""
    __tablename__ = "song_features"

    song_id = Column(Text, primary_key=True)
    mfcc = Column(ARRAY(Float))
    tempo = Column(Float)
    chroma = Column(ARRAY(Float))
    spectral_centroid = Column(Float)
    spectral_bandwidth = Column(Float)
    spectral_contrast = Column(ARRAY(Float))
    rms_energy = Column(Float)
    zcr = Column(Float)


class UserListeningHistory(Base):
    """User listening behavior for collaborative filtering."""
    __tablename__ = "user_listening_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text)
    song_id = Column(Text)
    listen_count = Column(Integer, default=1)
    last_played_at = Column(TIMESTAMP, default=datetime.utcnow)


class EmotionLabel(Base):
    """Emotion classifications for emotion-aware filtering."""
    __tablename__ = "emotion_labels"

    song_id = Column(Text, primary_key=True)
    emotion = Column(Text)


def get_db():
    """Create and return a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
