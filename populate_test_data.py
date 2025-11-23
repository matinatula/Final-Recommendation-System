# populate_test_data.py
import random
from database import engine, SessionLocal, Track, User, SongFeature, UserListeningHistory, EmotionLabel, create_tables

# Configuration
NUM_USERS = 500
NUM_SONGS = 1000
NUM_INTERACTIONS = 15000  # Realistic listening history

EMOTIONS = ["sad", "neutral", "happy", "fear", "angry"]

# Sample song name templates for variety
SONG_PREFIXES = ["Love", "Night", "Summer", "Winter", "Dance", "Dream", "Fire", "Rain", "Star", "Moon",
                 "Heart", "Soul", "Electric", "Golden", "Blue", "Red", "Sweet", "Wild", "Lost", "Free"]
SONG_SUFFIXES = ["Nights", "Dreams", "Vibes", "Feelings", "Memories", "Stories", "Symphony", "Anthem",
                 "Paradise", "Highway", "Thunder", "Lights", "Shadows", "Echoes", "Waves", "Beat"]

# Music genres for variety
GENRES = ["Pop", "Rock", "Hip Hop", "Jazz", "Electronic",
          "Classical", "R&B", "Country", "Metal", "Folk"]


def generate_song_name(index):
    """Generate a unique song name"""
    if index < 50:
        # First 50 are "real" famous songs
        famous_songs = [
            "Blinding Lights", "Shape of You", "Someone Like You", "Bohemian Rhapsody",
            "Happy", "Thriller", "Rolling in the Deep", "Uptown Funk", "Let It Be",
            "Smells Like Teen Spirit", "Imagine", "Billie Jean", "Hey Jude", "Sweet Child O Mine",
            "Lose Yourself", "Hotel California", "Stairway to Heaven", "Wonderwall", "Creep",
            "Radioactive", "Counting Stars", "Wake Me Up", "Roar", "Dark Horse", "All of Me",
            "Thinking Out Loud", "Sugar", "Shake It Off", "Blank Space", "Bad Blood",
            "Hello", "Sorry", "Love Yourself", "Can't Stop the Feeling", "Closer", "Stressed Out",
            "Ride", "Heathens", "Cheap Thrills", "Don't Let Me Down", "Cold Water", "Starboy",
            "Side to Side", "24K Magic", "Rockabye", "Black Beatles", "Believer", "Thunder",
            "Whatever It Takes", "Havana", "Perfect", "Too Good At Goodbyes"
        ]
        return famous_songs[index] if index < len(famous_songs) else f"Song {index + 1}"
    else:
        # Generate random song names
        prefix = random.choice(SONG_PREFIXES)
        suffix = random.choice(SONG_SUFFIXES)
        return f"{prefix} {suffix} {index - 49}"


def generate_songs(num_songs):
    """Generate song data"""
    songs = []
    for i in range(num_songs):
        songs.append({
            "id": f"track_{i+1:04d}",
            "name": generate_song_name(i),
            "duration_ms": random.randint(120000, 420000),  # 2-7 minutes
            "popularity": random.randint(30, 100),
        })
    return songs


def generate_users(num_users):
    """Generate user data"""
    users = []
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
                   "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose", "Sam", "Tina"]
    last_names = ["Smith", "Johnson", "Brown", "Davis", "Miller",
                  "Wilson", "Moore", "Taylor", "Anderson", "Thomas"]

    for i in range(num_users):
        first = random.choice(first_names)
        last = random.choice(last_names)
        users.append({
            "id": f"user_{i+1:04d}",
            "email": f"{first.lower()}.{last.lower()}{i}@example.com",
            "name": f"{first} {last}",
        })
    return users


def generate_audio_features(song_id):
    """Generate realistic random audio features for a song"""
    return {
        "song_id": song_id,
        "mfcc": [round(random.uniform(-50, 50), 2) for _ in range(13)],
        "tempo": round(random.uniform(80, 180), 2),
        "chroma": [round(random.uniform(0, 1), 3) for _ in range(12)],
        "spectral_centroid": round(random.uniform(1000, 4000), 2),
        "spectral_bandwidth": round(random.uniform(1500, 3000), 2),
        "spectral_contrast": [round(random.uniform(10, 30), 2) for _ in range(7)],
        "rms_energy": round(random.uniform(0.01, 0.3), 3),
        "zcr": round(random.uniform(0.01, 0.15), 4),
    }


def generate_listening_history(users, songs, num_interactions):
    """Generate realistic listening history"""
    history = []

    # Create realistic patterns: some users are heavy listeners, some light
    for user in users:
        # Each user listens to 10-50 songs
        num_user_songs = random.randint(10, 50)
        listened_songs = random.sample(songs, min(num_user_songs, len(songs)))

        for song in listened_songs:
            # Power law distribution: few songs played a lot, many played a little
            if random.random() < 0.2:  # 20% are "favorite" songs
                listen_count = random.randint(20, 100)
            else:
                listen_count = random.randint(1, 10)

            history.append({
                "user_id": user["id"],
                "song_id": song["id"],
                "listen_count": listen_count,
            })

            # Stop if we've reached target interactions
            if len(history) >= num_interactions:
                return history

    return history


def populate_database():
    """Main function to populate the database with test data"""

    print("🚀 Starting database population...")
    print(
        f"📊 Target: {NUM_USERS} users, {NUM_SONGS} songs, ~{NUM_INTERACTIONS} interactions")
    print("⏳ This will take about 2 minutes...\n")

    # Generate all data first
    print("🎲 Generating data...")
    songs = generate_songs(NUM_SONGS)
    users = generate_users(NUM_USERS)
    print(f"   ✅ Generated {len(songs)} songs and {len(users)} users")

    # Step 1: Create all tables
    print("\n📋 Creating tables...")
    create_tables()

    # Step 2: Create a database session
    db = SessionLocal()

    try:
        # Step 3: Clear existing data
        print("\n🧹 Clearing existing data...")
        db.query(UserListeningHistory).delete()
        db.query(EmotionLabel).delete()
        db.query(SongFeature).delete()
        db.query(Track).delete()
        db.query(User).delete()
        db.commit()
        print("   ✅ Old data cleared")

        # Step 4: Add tracks (in batches for speed)
        print(f"\n🎵 Adding {NUM_SONGS} tracks...")
        batch_size = 100
        for i in range(0, len(songs), batch_size):
            batch = songs[i:i+batch_size]
            for song_data in batch:
                track = Track(**song_data)
                db.add(track)
            db.commit()
            print(
                f"   ⏳ Added {min(i+batch_size, len(songs))}/{len(songs)} tracks...", end='\r')
        print(f"\n   ✅ Added {len(songs)} tracks")

        # Step 5: Add users
        print(f"\n👥 Adding {NUM_USERS} users...")
        for i in range(0, len(users), batch_size):
            batch = users[i:i+batch_size]
            for user_data in batch:
                user = User(**user_data)
                db.add(user)
            db.commit()
            print(
                f"   ⏳ Added {min(i+batch_size, len(users))}/{len(users)} users...", end='\r')
        print(f"\n   ✅ Added {len(users)} users")

        # Step 6: Add song features
        print(f"\n🎼 Adding audio features for {NUM_SONGS} songs...")
        for i, song in enumerate(songs):
            features = generate_audio_features(song["id"])
            song_feature = SongFeature(**features)
            db.add(song_feature)

            if (i + 1) % batch_size == 0:
                db.commit()
                print(
                    f"   ⏳ Added {i+1}/{len(songs)} audio features...", end='\r')
        db.commit()
        print(f"\n   ✅ Added audio features for {len(songs)} songs")

        # Step 7: Add emotion labels
        print(f"\n😄 Adding emotion labels for {NUM_SONGS} songs...")
        for i, song in enumerate(songs):
            emotion = EmotionLabel(
                song_id=song["id"],
                emotion=random.choice(EMOTIONS)
            )
            db.add(emotion)

            if (i + 1) % batch_size == 0:
                db.commit()
                print(
                    f"   ⏳ Added {i+1}/{len(songs)} emotion labels...", end='\r')
        db.commit()
        print(f"\n   ✅ Added emotion labels for {len(songs)} songs")

        # Step 8: Add listening history
        print(f"\n📊 Generating and adding listening history...")
        history_data = generate_listening_history(
            users, songs, NUM_INTERACTIONS)
        for i, history_item in enumerate(history_data):
            listening = UserListeningHistory(**history_item)
            db.add(listening)

            if (i + 1) % batch_size == 0:
                db.commit()
                print(
                    f"   ⏳ Added {i+1}/{len(history_data)} listening records...", end='\r')
        db.commit()
        print(f"\n   ✅ Added {len(history_data)} listening history records")

        print("\n" + "="*60)
        print("✅ DATABASE POPULATED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📊 Final Summary:")
        print(f"   • {len(songs):,} songs")
        print(f"   • {len(users):,} users")
        print(f"   • {len(songs):,} audio feature sets")
        print(f"   • {len(songs):,} emotion labels")
        print(f"   • {len(history_data):,} listening records")
        print(f"\n🎉 Your recommendation system has realistic data!")
        print("🚀 Ready to build your recommender modules!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    populate_database()
