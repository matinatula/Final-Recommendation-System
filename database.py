import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

load_dotenv()

# Database connection
engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """User account information - matches Drizzle 'user' table"""

    __tablename__ = "user"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False, unique=True)
    email_verified = Column("email_verified", Boolean,
                            default=False, nullable=False)
    image = Column(Text)
    created_at = Column(
        "created_at", TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        "updated_at",
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Track(Base):
    """Track metadata - matches Drizzle 'tracks' table"""

    __tablename__ = "tracks"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    track_number = Column("track_number", Integer)
    duration_ms = Column("duration_ms", Integer, nullable=False)
    explicit = Column(Boolean, nullable=False)
    popularity = Column(Integer)
    preview_url = Column("preview_url", Text)
    external_urls = Column("external_urls", JSONB, nullable=False)
    artists = Column(JSONB, nullable=False)
    created_at = Column(
        "created_at", TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        "updated_at",
        TIMESTAMP(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Foreign keys
    album_id = Column("album_id", Text, ForeignKey(
        "albums.id", ondelete="CASCADE"))
    top_tracks_artist_id = Column(
        "top_tracks_artist_id", Text, ForeignKey(
            "artists.id", ondelete="CASCADE")
    )
    stream_url = Column("stream_url", Text)


class TrackFeature(Base):
    """Audio features for content-based filtering - matches Drizzle 'track_features' table"""

    __tablename__ = "track_features"

    song_id = Column(
        "song_id", Text, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    mfcc = Column(ARRAY(Float), nullable=False)
    tempo = Column(Float, nullable=False)
    chroma = Column(ARRAY(Float), nullable=False)
    spectral_centroid = Column("spectral_centroid", Float, nullable=False)
    spectral_bandwidth = Column("spectral_bandwidth", Float, nullable=False)
    spectral_contrast = Column(
        "spectral_contrast", ARRAY(Float), nullable=False)
    rms_energy = Column("rms_energy", Float, nullable=False)
    zcr = Column(Float, nullable=False)


class EmotionLabel(Base):
    """Emotion classifications - matches Drizzle 'emotion_labels' table"""

    __tablename__ = "emotion_labels"

    song_id = Column(
        "song_id", Text, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    emotion = Column(Text, nullable=False)


class UserListeningHistory(Base):
    """User listening behavior for collaborative filtering"""

    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        "user_id", Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    song_id = Column(
        "track_id", Text, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    listen_count = Column("listen_count", Integer, default=1)


class UserRecommendation(Base):
    """User recommendations - matches Drizzle 'user_recommendations' table"""

    __tablename__ = "user_recommendations"

    id = Column(Text, primary_key=True)
    user_id = Column(
        "user_id", Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    track_id = Column(
        "track_id", Text, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    score = Column(Float, nullable=False)
    reason = Column(Text)
    created_at = Column(
        "created_at", TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )


# Legacy table name alias for backwards compatibility
# If your old code uses SongFeature, it will still work
SongFeature = TrackFeature


def get_db():
    """Create and return a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


if __name__ == "__main__":
    # Test database connection
    try:
        with engine.connect() as connection:
            print("Database connection successful!")
            print(f"Connected to: {engine.url.database}")
    except Exception as e:
        print(f"Database connection failed: {e}")
