"""
Deep diagnostic to see what's ACTUALLY in the database.
"""

import numpy as np
from database import SessionLocal, UserListeningHistory, SongFeature

print("="*60)
print("🔍 DEEP DIAGNOSTIC - DATABASE INSPECTION")
print("="*60)

db = SessionLocal()

# Check 1: How many users have listening history?
all_users = db.query(UserListeningHistory.user_id).distinct().all()
all_user_ids = [u[0] for u in all_users]
print(f"\n👥 Total users: {len(all_user_ids)}")

# Check 2: Pick a user and see their songs
if all_user_ids:
    user_id = all_user_ids[0]
    user_history = db.query(UserListeningHistory).filter(
        UserListeningHistory.user_id == user_id
    ).all()

    print(f"\n📻 User {user_id} listening history:")
    print(f"   Total unique songs: {len(user_history)}")

    # Extract song IDs
    user_song_ids = [h.song_id for h in user_history]
    song_nums = [int(s.split('_')[1]) for s in user_song_ids]
    song_styles = [n % 5 for n in song_nums]

    print(f"   Song IDs: {user_song_ids[:5]}... (showing first 5)")
    print(f"   Song numbers: {song_nums[:5]}...")
    print(f"   Song styles: {song_styles[:5]}...")

    # Check if all songs are same style
    unique_styles = set(song_styles)
    print(f"   ❓ Unique styles in this user's history: {unique_styles}")
    if len(unique_styles) == 1:
        print(
            f"   ✅ GOOD: User only listens to style {list(unique_styles)[0]}")
    else:
        print(f"   ❌ PROBLEM: User listens to mixed styles!")

# Check 3: Are songs in same style actually similar?
print(f"\n🎵 Checking feature similarity for songs in same style:")

# Get all songs
all_features = db.query(SongFeature).all()
print(f"   Total songs in database: {len(all_features)}")

# Group by style
songs_by_style = {0: [], 1: [], 2: [], 3: [], 4: []}
for f in all_features:
    track_num = int(f.song_id.split('_')[1])
    style = track_num % 5
    songs_by_style[style].append(f)

# Check a few songs from same style
for style in range(5):
    style_songs = songs_by_style[style]
    if len(style_songs) >= 2:
        s1 = style_songs[0]
        s2 = style_songs[1]

        # Compare MFCC
        mfcc1 = np.array(s1.mfcc)
        mfcc2 = np.array(s2.mfcc)

        # Simple distance
        mfcc_distance = np.linalg.norm(mfcc1 - mfcc2)

        print(f"   Style {style}: {s1.song_id} vs {s2.song_id}")
        print(f"      MFCC distance: {mfcc_distance:.4f}")
        print(f"      Tempo: {s1.tempo:.1f} vs {s2.tempo:.1f}")

# Check 4: Train/Test split
print(f"\n📊 Train/Test split check:")
if all_user_ids:
    user_id = all_user_ids[0]
    history = db.query(UserListeningHistory).filter(
        UserListeningHistory.user_id == user_id
    ).all()

    if history:
        history_sorted = sorted(
            history, key=lambda x: x.listen_count, reverse=True)
        split_point = max(1, int(len(history_sorted) * 0.8))  # 80/20 split

        train_songs = history_sorted[:split_point]
        test_songs = history_sorted[split_point:]

        train_ids = [h.song_id for h in train_songs]
        test_ids = [h.song_id for h in test_songs]

        print(f"   Train songs ({len(train_songs)}): {train_ids[:5]}...")
        print(f"   Test songs ({len(test_songs)}): {test_ids[:5]}...")

        # Check: Are they from same style?
        train_styles = [int(s.split('_')[1]) % 5 for s in train_ids]
        test_styles = [int(s.split('_')[1]) % 5 for s in test_ids]

        print(f"   Train styles: {set(train_styles)}")
        print(f"   Test styles: {set(test_styles)}")

        if set(train_styles) == set(test_styles):
            print(f"   ✅ GOOD: Train and test from same style")
        else:
            print(f"   ❌ PROBLEM: Train and test from different styles!")

db.close()
print("\n" + "="*60)
