# debug_collaborative.py
# Figure out why Collaborative NDCG = 0

from collections import Counter
from database import SessionLocal, UserListeningHistory
from collaborative_als import CollaborativeRecommender

db = SessionLocal()

# Pick a test user
user_id = 'user_0001'

# Get user's listening history
history = db.query(UserListeningHistory)\
    .filter(UserListeningHistory.user_id == user_id)\
    .order_by(UserListeningHistory.listen_count.desc())\
    .all()

print(f"User {user_id} listening history: {len(history)} songs")

# Split into train/test (same as NDCG evaluator)
split_point = int(len(history) * 0.8)
train_songs = set(h.song_id for h in history[:split_point])
test_songs = set(h.song_id for h in history[split_point:])

print(f"Train songs ({len(train_songs)}): {list(train_songs)[:5]}...")
print(f"Test songs ({len(test_songs)}): {list(test_songs)}")

# Get test song styles
test_styles = [int(s.replace('track_', '')) % 5 for s in test_songs]
print(f"Test song styles: {test_styles}")

# Now get collaborative recommendations
print("\n" + "="*50)
print("Getting Collaborative Recommendations...")
print("="*50)

collab = CollaborativeRecommender()
collab.load_data()
collab.train()

recs = collab.recommend_for_user(user_id, top_n=50)

print(f"\nGot {len(recs)} recommendations")
print(f"Top 10 recommended songs:")
for i, r in enumerate(recs[:10], 1):
    song_num = int(r['song_id'].replace('track_', ''))
    style = song_num % 5
    in_test = "✅ IN TEST!" if r['song_id'] in test_songs else ""
    print(
        f"  {i}. {r['song_id']} (style {style}, score {r['score']:.4f}) {in_test}")

# Check overlap
rec_ids = set(r['song_id'] for r in recs)
overlap = rec_ids & test_songs
print(f"\n🎯 Overlap with test set: {len(overlap)} songs")
print(f"   Test songs: {test_songs}")
print(f"   Overlap: {overlap}")

# Check if test songs are in the recommendation candidate pool at all
print("\n" + "="*50)
print("Checking if test songs could even be recommended...")
print("="*50)

for test_song in test_songs:
    if test_song in collab.song_id_to_index:
        idx = collab.song_id_to_index[test_song]
        print(f"  {test_song}: index {idx} ✅ (can be recommended)")
    else:
        print(f"  {test_song}: NOT IN INDEX ❌ (cannot be recommended!)")

# Check recommended song styles
print("\n" + "="*50)
print("Style distribution in recommendations:")
print("="*50)
rec_styles = [int(r['song_id'].replace('track_', '')) % 5 for r in recs]
style_counts = Counter(rec_styles)
user_style = int(list(train_songs)[0].replace('track_', '')) % 5
print(f"User's style (from train): {user_style}")
print(f"Recommended styles: {dict(style_counts)}")

if style_counts.get(user_style, 0) > len(recs) * 0.5:
    print("✅ GOOD: Collaborative is recommending user's preferred style!")
else:
    print("⚠️  Collaborative is NOT focusing on user's preferred style")

collab.close()
db.close()
