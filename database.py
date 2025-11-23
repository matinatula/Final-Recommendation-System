# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, Text, TIMESTAMP, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get database credentials from .env
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Create database connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create database engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


# ========== TABLE MODELS ==========

class Track(Base):
    """Tracks table - existing in your schema"""
    __tablename__ = "tracks"

    id = Column(Text, primary_key=True)
    name = Column(Text)
    duration_ms = Column(Integer)
    popularity = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class User(Base):
    """Users table - existing in your schema"""
    __tablename__ = "users"

    id = Column(Text, primary_key=True)
    email = Column(Text)
    name = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class SongFeature(Base):
    """Song features table - NEW (you need this for content-based)"""
    __tablename__ = "song_features"

    song_id = Column(Text, primary_key=True)
    mfcc = Column(ARRAY(Float))  # MFCC coefficients
    tempo = Column(Float)
    chroma = Column(ARRAY(Float))
    spectral_centroid = Column(Float)
    spectral_bandwidth = Column(Float)
    spectral_contrast = Column(ARRAY(Float))
    rms_energy = Column(Float)
    zcr = Column(Float)


class UserListeningHistory(Base):
    """User listening history - NEW (you need this for ALS)"""
    __tablename__ = "user_listening_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text)
    song_id = Column(Text)
    listen_count = Column(Integer, default=1)
    last_played_at = Column(TIMESTAMP, default=datetime.utcnow)


class EmotionLabel(Base):
    """Emotion labels - NEW (you need this for emotion filtering)"""
    __tablename__ = "emotion_labels"

    song_id = Column(Text, primary_key=True)
    emotion = Column(Text)  # 'sad', 'neutral', 'happy', 'fear', 'angry'


# Function to get a database session
def get_db():
    """Returns a database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


# Function to create all tables
def create_tables():
    """Creates all tables in the database"""
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully!")
