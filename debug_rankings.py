# debug_rankings.py
# Check where test songs rank in recommendations

import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from database import SessionLocal, UserListeningHistory, SongFeature

db = SessionLocal()

# Get one user
user_id = 'user_0001'
history = db.query(UserListeningHistory)\
    .filter(UserListeningHistory.user_id == user_id)\
    .order_by(UserListeningHistory.listen_count.desc()).all()

split = int(len(history) * 0.8)
train_songs = {h.song_id: h.listen_count for h in history[:split]}
test_songs = {h.song_id for h in history[split:]}

print(f"User {user_id}: {len(train_songs)} train, {len(test_songs)} test")
print(f"Test songs: {test_songs}")

# Load features
features = db.query(SongFeature).all()
song_ids = [f.song_id for f in features]
feature_list = []
for f in features:
    vec = f.mfcc + [f.tempo] + f.chroma + [f.spectral_centroid,
                                           f.spectral_bandwidth] + f.spectral_contrast + [f.rms_energy, f.zcr]
    feature_list.append(vec)
feature_matrix = StandardScaler().fit_transform(np.array(feature_list))
song_to_idx = {s: i for i, s in enumerate(song_ids)}

# Content-based: Get rankings for ALL songs
train_indices = [song_to_idx[s] for s in train_songs if s in song_to_idx]
user_profile = np.mean(feature_matrix[train_indices], axis=0).reshape(1, -1)
similarities = cosine_similarity(user_profile, feature_matrix)[0]

# Don't filter anything - get raw rankings
ranking = np.argsort(similarities)[::-1]
song_ranks = {song_ids[idx]: rank+1 for rank, idx in enumerate(ranking)}

print(f"\n📊 CONTENT-BASED: Where do test songs rank?")
print(f"   (Out of {len(song_ids)} total songs)")
for song in test_songs:
    rank = song_ranks.get(song, "N/A")
    sim = similarities[song_to_idx[song]] if song in song_to_idx else 0
    print(f"   {song}: rank {rank}, similarity {sim:.4f}")

# Check: What's the style of test songs vs top recommended?
test_styles = [int(s.replace('track_', '')) % 5 for s in test_songs]
print(f"\nTest song styles: {set(test_styles)}")

top_10_songs = [song_ids[ranking[i]] for i in range(10)]
top_10_styles = [int(s.replace('track_', '')) % 5 for s in top_10_songs]
print(f"Top 10 recommended styles: {top_10_styles}")

# KEY QUESTION: Are test songs among top songs of their style?
user_style = test_styles[0]
same_style_songs = [s for s in song_ids if int(
    s.replace('track_', '')) % 5 == user_style]
same_style_sims = [(s, similarities[song_to_idx[s]]) for s in same_style_songs]
same_style_sims.sort(key=lambda x: x[1], reverse=True)

print(f"\n📊 Within STYLE {user_style} only ({len(same_style_songs)} songs):")
print(f"Top 10 style-{user_style} songs:")
for i, (s, sim) in enumerate(same_style_sims[:10], 1):
    marker = "⭐ TEST" if s in test_songs else ""
    print(f"   {i}. {s}: {sim:.4f} {marker}")

print(f"\nWhere do test songs rank WITHIN their style?")
style_ranking = {s: i+1 for i, (s, _) in enumerate(same_style_sims)}
for song in test_songs:
    rank = style_ranking.get(song, "N/A")
    print(f"   {song}: rank {rank}/{len(same_style_songs)}")

db.close()
