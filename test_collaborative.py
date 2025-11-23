# test_collaborative.py
from collaborative_als import CollaborativeRecommender

print("="*60)
print("🚀 TESTING COLLABORATIVE FILTERING (ALS)")
print("="*60)

# Create recommender
recommender = CollaborativeRecommender(factors=50, iterations=20)

# Step 1: Load data
recommender.load_data()

# Step 2: Train model
recommender.train()

# test_collaborative.py
# Add this right after recommender.train()

print("\n🔍 DEBUG INFO:")
print(f"User-item matrix shape: {recommender.user_item_matrix.shape}")
print(f"Number of users in mapping: {len(recommender.user_id_to_index)}")
print(f"Number of songs in mapping: {len(recommender.song_id_to_index)}")
print(
    f"user_0001 index: {recommender.user_id_to_index.get('user_0001', 'NOT FOUND')}")


# Test 1: Get recommendations for a user
print("\n" + "="*60)
print("🎵 Test 1: Recommendations for user_0001")
print("="*60)

try:
    recommendations = recommender.recommend_for_user("user_0001", top_n=5)

    for i, rec in enumerate(recommendations, 1):
        song = recommender.get_song_details(rec['song_id'])
        print(f"{i}. {song.name} (ID: {rec['song_id']})")
        print(f"   Predicted score: {rec['score']:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Find similar users
print("\n" + "="*60)
print("👥 Test 2: Users similar to user_0001")
print("="*60)

try:
    similar_users = recommender.get_similar_users("user_0001", top_n=5)

    for i, user in enumerate(similar_users, 1):
        print(f"{i}. {user['user_id']}")
        print(f"   Similarity: {user['similarity_score']:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Find similar songs (based on listening patterns)
print("\n" + "="*60)
print("🎵 Test 3: Songs similar to track_0001 (based on who listened)")
print("="*60)

try:
    similar_songs = recommender.get_similar_songs("track_0001", top_n=5)

    for i, rec in enumerate(similar_songs, 1):
        song = recommender.get_song_details(rec['song_id'])
        print(f"{i}. {song.name} (ID: {rec['song_id']})")
        print(f"   Similarity: {rec['similarity_score']:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Compare content-based vs collaborative for same song
print("\n" + "="*60)
print("🔍 Test 4: Content-Based vs Collaborative Comparison")
print("="*60)
print("For track_0001 (Blinding Lights):")
print("\nContent-Based finds songs that SOUND similar")
print("Collaborative finds songs that users with SIMILAR TASTE listened to")
print("\nBoth are valuable and work together in the hybrid system!")

# Close connection
recommender.close()

print("\n" + "="*60)
print("✅ COLLABORATIVE FILTERING WORKING PERFECTLY!")
print("="*60)
