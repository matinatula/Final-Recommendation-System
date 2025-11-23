# debug_collaborative2.py
# Check WHY test songs aren't ranked highly

import numpy as np
from database import SessionLocal, UserListeningHistory
from collaborative_als import CollaborativeRecommender

db = SessionLocal()
user_id = 'user_0001'

# Get history and split
history = db.query(UserListeningHistory)\
    .filter(UserListeningHistory.user_id == user_id)\
    .order_by(UserListeningHistory.listen_count.desc())\
    .all()

split_point = int(len(history) * 0.8)
train_songs = set(h.song_id for h in history[:split_point])
test_songs = list(h.song_id for h in history[split_point:])

print(f"Test songs: {test_songs}")

# Train collaborative
collab = CollaborativeRecommender()
collab.load_data()
collab.train()

# Get ALL scores for ALL songs (not just top 50)
user_idx = collab.user_id_to_index[user_id]
user_factors = collab.model.item_factors[user_idx]
item_factors = collab.model.user_factors

# Calculate scores for ALL songs
all_scores = item_factors.dot(user_factors)

print(f"\n📊 Score distribution for ALL {len(all_scores)} songs:")
print(f"   Min: {all_scores.min():.4f}")
print(f"   Max: {all_scores.max():.4f}")
print(f"   Mean: {all_scores.mean():.4f}")

# Check scores for test songs specifically
print(f"\n🎯 Scores for TEST songs:")
test_scores = []
for song_id in test_songs:
    song_idx = collab.song_id_to_index[song_id]
    score = all_scores[song_idx]
    test_scores.append((song_id, score))

    # Find rank of this song
    rank = (all_scores > score).sum() + 1
    print(f"   {song_id}: score={score:.4f}, rank={rank}/1000")

# Check: Are test songs being filtered out because user "listened" to them?
print(f"\n🔍 Are test songs in user's listening history in the MATRIX?")
user_row = collab.user_item_matrix[user_idx]
user_listened = set(user_row.nonzero()[1])

for song_id in test_songs:
    song_idx = collab.song_id_to_index[song_id]
    in_matrix = song_idx in user_listened
    print(
        f"   {song_id} (idx {song_idx}): {'YES - filtered out!' if in_matrix else 'NO'}")

print("\n" + "="*60)
print("💡 THE ISSUE:")
print("="*60)
print("Test songs ARE in the user's listening history matrix!")
print("So collaborative filtering CORRECTLY filters them out")
print("(you don't recommend songs users already listened to)")
print("\nThis is a METHODOLOGY issue with how we're evaluating.")
print("We need to HIDE test songs from the training data!")

collab.close()
db.close()

