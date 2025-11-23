"""
Detailed matching debug - see exactly what's happening with similarity scores.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from database import SessionLocal, UserListeningHistory, SongFeature

print("="*60)
print("🔍 DETAILED MATCHING DEBUG")
print("="*60)

db = SessionLocal()

# Load all features (same as content_based.py)
features = db.query(SongFeature).all()
song_ids = [f.song_id for f in features]

feature_list = []
for f in features:
    feature_vector = (
        f.mfcc + [f.tempo] + f.chroma + [f.spectral_centroid] +
        [f.spectral_bandwidth] + f.spectral_contrast + [f.rms_energy] + [f.zcr]
    )
    feature_list.append(feature_vector)

feature_matrix = np.array(feature_list)
scaler = StandardScaler()
feature_matrix = scaler.fit_transform(feature_matrix)

print(
    f"✅ Loaded {len(song_ids)} songs with {feature_matrix.shape[1]} features")

# Get a test user
all_users = db.query(UserListeningHistory.user_id).distinct().all()
user_id = all_users[0][0]

user_history = db.query(UserListeningHistory).filter(
    UserListeningHistory.user_id == user_id
).all()

# Sort and split (same as NDCG does)
history_sorted = sorted(
    user_history, key=lambda x: x.listen_count, reverse=True)
split_point = max(1, int(len(history_sorted) * 0.8))

train_songs = [h.song_id for h in history_sorted[:split_point]]
test_songs = [h.song_id for h in history_sorted[split_point:]]

print(f"\n👤 User: {user_id}")
print(f"   Train songs: {train_songs}")
print(f"   Test songs: {test_songs}")

# Get user profile (average of train songs)
train_indices = [song_ids.index(s) for s in train_songs if s in song_ids]
user_profile = np.mean(feature_matrix[train_indices], axis=0).reshape(1, -1)

# Calculate similarity with ALL songs
similarities = cosine_similarity(user_profile, feature_matrix)[0]

# Exclude train songs
for idx in train_indices:
    similarities[idx] = -1

# Get top 10
top_indices = np.argsort(similarities)[::-1][:10]
recommendations = [song_ids[i] for i in top_indices]
rec_scores = [similarities[i] for i in top_indices]

print(f"\n🎯 Top 10 recommendations:")
for i, (rec, score) in enumerate(zip(recommendations, rec_scores), 1):
    in_test = "✅" if rec in test_songs else "❌"
    print(f"   {i}. {rec} (score: {score:.4f}) {in_test}")

# KEY INSIGHT: Check similarity between train and test songs
print(f"\n🔍 KEY INSIGHT - Similarity between train and test songs:")

for test_song in test_songs[:3]:
    if test_song in song_ids:
        test_idx = song_ids.index(test_song)
        test_vector = feature_matrix[test_idx].reshape(1, -1)

        # How similar is this test song to user profile?
        test_sim_to_profile = cosine_similarity(
            test_vector, user_profile)[0][0]

        # How similar is it to train songs?
        sims_to_train = cosine_similarity(
            test_vector, feature_matrix[train_indices])
        max_sim_to_train = np.max(sims_to_train)

        print(f"   {test_song}:")
        print(f"      Similarity to user profile: {test_sim_to_profile:.4f}")
        print(
            f"      Max similarity to any train song: {max_sim_to_train:.4f}")
        print(f"      Rank in recommendations: ", end="")

        if test_song in recommendations:
            rank = recommendations.index(test_song) + 1
            print(f"#{rank}")
        else:
            # Find its rank
            test_rank = np.argsort(similarities)[::-1]
            test_rank_pos = np.where(test_rank == test_idx)[0]
            if len(test_rank_pos) > 0:
                print(f"#{test_rank_pos[0]+1} (outside top 10)")
            else:
                print(f"NOT FOUND")

# Summary stats
print(f"\n📊 Similarity Score Statistics:")
valid_sims = similarities[similarities > -1]
print(f"   Mean similarity: {np.mean(valid_sims):.4f}")
print(f"   Std similarity: {np.std(valid_sims):.4f}")
print(f"   Min: {np.min(valid_sims):.4f}")
print(f"   Max: {np.max(valid_sims):.4f}")

# Check: Are test songs similar to train songs at all?
print(f"\n🔎 Are test songs in the database?")
for test_song in test_songs:
    if test_song in song_ids:
        print(f"   ✅ {test_song} found")
    else:
        print(f"   ❌ {test_song} NOT found")

db.close()
print("\n" + "="*60)
