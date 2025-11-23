"""
Diagnostic script to understand what content-based is doing.
Run this SEPARATELY to debug.
"""

import numpy as np
from content_based import ContentBasedRecommender
from database import SessionLocal, UserListeningHistory

print("="*60)
print("🔍 CONTENT-BASED DIAGNOSTIC")
print("="*60)

# Initialize
recommender = ContentBasedRecommender()
recommender.load_features()
db = SessionLocal()

# Pick a random user
all_users = db.query(UserListeningHistory.user_id).distinct().all()
user_id = all_users[0][0]  # First user

print(f"\n👤 Testing with user: {user_id}")

# Get their listening history
user_history = db.query(UserListeningHistory)\
    .filter(UserListeningHistory.user_id == user_id)\
    .all()

user_song_ids = [h.song_id for h in user_history]
print(f"   Listened to {len(user_song_ids)} songs")

# Show first few songs they listened to
print(f"\n📻 First 5 songs they listened to:")
for song_id in user_song_ids[:5]:
    track = recommender.get_song_details(song_id)
    if track:
        print(f"   ✓ {song_id}: {track.name}")

# Get recommendations
print(f"\n🎵 Getting recommendations for this user...")
recommendations = recommender.recommend_for_user_history(
    user_song_ids=user_song_ids,
    top_n=10
)

print(f"\n🎯 Top 10 recommendations:")
for i, rec in enumerate(recommendations, 1):
    track = recommender.get_song_details(rec['song_id'])
    score = rec['score']
    if track:
        print(f"   {i}. {rec['song_id']}: {track.name} (score: {score:.4f})")
    else:
        print(f"   {i}. {rec['song_id']}: UNKNOWN (score: {score:.4f})")

# KEY CHECK: Do any recommendations overlap with their listening history?
print(f"\n🔍 KEY CHECK:")
recommended_ids = [rec['song_id'] for rec in recommendations]
overlap = set(recommended_ids) & set(user_song_ids)
print(f"   Overlap between recommendations and history: {len(overlap)}")
if overlap:
    print(f"   ⚠️  Recommended songs they already heard: {overlap}")
else:
    print(f"   ✅ No overlap (correct - we excluded their songs)")

# Check the similarity scores
print(f"\n📊 Similarity score range:")
scores = [rec['score'] for rec in recommendations]
print(f"   Min: {min(scores):.4f}")
print(f"   Max: {max(scores):.4f}")
print(f"   Mean: {np.mean(scores):.4f}")

if max(scores) < 0.3:
    print(f"   ⚠️  WARNING: Similarity scores are very LOW!")
    print(f"       This means recommendations are DISSIMILAR to user's taste")

print("\n" + "="*60)

recommender.close()
db.close()
