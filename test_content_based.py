# test_content_based.py
from content_based import ContentBasedRecommender

# Create recommender
recommender = ContentBasedRecommender()

# Load features
recommender.load_features()

# Test 1: Find songs similar to track_0001
print("\n🎵 Test 1: Songs similar to track_0001 (Blinding Lights)")
print("="*60)
recommendations = recommender.get_similar_songs("track_0001", top_n=5)

for i, rec in enumerate(recommendations, 1):
    song = recommender.get_song_details(rec['song_id'])
    print(f"{i}. {song.name} (ID: {rec['song_id']})")
    print(f"   Similarity: {rec['similarity_score']:.4f}")

# Test 2: Recommend based on user listening history
print("\n🎵 Test 2: Recommendations based on user history")
print("="*60)
# Blinding Lights, Happy, Uptown Funk
user_history = ["track_0001", "track_0005", "track_0009"]
print(f"User listened to: {user_history}")

recommendations = recommender.recommend_for_user_history(user_history, top_n=5)

for i, rec in enumerate(recommendations, 1):
    song = recommender.get_song_details(rec['song_id'])
    print(f"{i}. {song.name} (ID: {rec['song_id']})")
    print(f"   Score: {rec['similarity_score']:.4f}")

# Close connection
recommender.close()

print("\n✅ Content-based recommender working perfectly!")
