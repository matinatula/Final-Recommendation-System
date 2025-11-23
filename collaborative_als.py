# collaborative_als.py
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from database import SessionLocal, UserListeningHistory, Track


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using implicit ALS.
    Learns from user listening patterns to predict what users might like.
    """

    def __init__(self, factors=50, regularization=0.01, iterations=20):
        """Initialize the collaborative recommender."""
        self.db = SessionLocal()
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )

        self.user_item_matrix = None
        self.item_user_matrix = None  # Store this separately
        self.user_id_to_index = {}
        self.index_to_user_id = {}
        self.song_id_to_index = {}
        self.index_to_song_id = {}

        self.is_trained = False

    def load_data(self):
        """Load listening history from database and build user-item matrix."""
        print("📥 Loading listening history from database...")

        history = self.db.query(UserListeningHistory).all()

        if not history:
            raise ValueError("❌ No listening history found in database!")

        print(f"✅ Found {len(history)} listening records")

        unique_users = sorted(list(set([h.user_id for h in history])))
        unique_songs = sorted(list(set([h.song_id for h in history])))

        print(f"✅ Unique users: {len(unique_users)}")
        print(f"✅ Unique songs: {len(unique_songs)}")

        # Create bidirectional mappings
        self.user_id_to_index = {user_id: idx for idx,
                                 user_id in enumerate(unique_users)}
        self.index_to_user_id = {idx: user_id for idx,
                                 user_id in enumerate(unique_users)}

        self.song_id_to_index = {song_id: idx for idx,
                                 song_id in enumerate(unique_songs)}
        self.index_to_song_id = {idx: song_id for idx,
                                 song_id in enumerate(unique_songs)}

        rows = []
        cols = []
        data = []

        for h in history:
            user_idx = self.user_id_to_index[h.user_id]
            song_idx = self.song_id_to_index[h.song_id]
            rows.append(user_idx)
            cols.append(song_idx)
            data.append(h.listen_count)

        # User-item matrix: users × songs
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_songs)),
            dtype=np.float32
        )

        print(f"✅ Built user-item matrix: {self.user_item_matrix.shape}")
        print(
            f"   (rows={len(unique_users)} users, cols={len(unique_songs)} songs)")

    def train(self):
        """Train the ALS model."""
        if self.user_item_matrix is None:
            raise ValueError("❌ Must load data first!")

        print("\n🧠 Training ALS model...")
        print("   This learns hidden patterns in user preferences...")

        # implicit library expects item-user matrix for .fit()
        # Store this for later use
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        self.model.fit(self.item_user_matrix)

        self.is_trained = True
        print("✅ Model trained successfully!")

    def recommend_for_user(self, user_id, top_n=10):
        """
        Get recommendations for a specific user.
        Uses manual scoring to avoid indexing issues.
        """
        if not self.is_trained:
            raise ValueError("❌ Model not trained!")

        if user_id not in self.user_id_to_index:
            raise ValueError(f"❌ User {user_id} not found!")

        user_idx = self.user_id_to_index[user_id]

        # Validate index
        if user_idx >= self.user_item_matrix.shape[0]:
            raise ValueError(
                f"❌ User index {user_idx} out of bounds (max: {self.user_item_matrix.shape[0]-1})")

        # Get user's listening history to filter out
        user_row = self.user_item_matrix[user_idx]
        user_songs = set(user_row.nonzero()[1])

        # IMPORTANT: After fitting, implicit stores:
        # - user_factors: shape (n_items, n_factors) because we transposed
        # - item_factors: shape (n_users, n_factors) because we transposed
        # This is counterintuitive but correct!

        # Since we fit with item_user_matrix (songs × users),
        # the model thinks items are songs and users are users (swapped)
        # So we need to use item_factors for our user
        user_factors = self.model.item_factors[user_idx]
        item_factors = self.model.user_factors

        # Calculate scores for all songs
        scores = item_factors.dot(user_factors)

        # Filter out already listened songs
        for song_idx in user_songs:
            scores[song_idx] = -np.inf

        # Get top N
        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            if scores[idx] > -np.inf:
                song_id = self.index_to_song_id[int(idx)]
                recommendations.append({
                    'song_id': song_id,
                    'score': float(scores[idx])
                })

        return recommendations

    def get_similar_users(self, user_id, top_n=10):
        """Find users with similar taste."""
        if not self.is_trained:
            raise ValueError("❌ Model not trained!")

        if user_id not in self.user_id_to_index:
            raise ValueError(f"❌ User {user_id} not found!")

        user_idx = self.user_id_to_index[user_id]

        # Validate index
        if user_idx >= self.user_item_matrix.shape[0]:
            raise ValueError(
                f"❌ User index {user_idx} out of bounds (max: {self.user_item_matrix.shape[0]-1})")

        # Get user factor (remember the swap!)
        user_factor = self.model.item_factors[user_idx]

        # Calculate similarity with all users
        similarities = self.model.item_factors.dot(user_factor)

        # Get top N (excluding the user themselves)
        similarities[user_idx] = -np.inf
        top_indices = np.argsort(similarities)[::-1][:top_n]

        similar_users = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                similar_user_id = self.index_to_user_id[int(idx)]
                similar_users.append({
                    'user_id': similar_user_id,
                    'similarity_score': float(similarities[idx])
                })

        return similar_users

    def get_similar_songs(self, song_id, top_n=10):
        """Find songs similar to the given song (based on user behavior)."""
        if not self.is_trained:
            raise ValueError("❌ Model not trained!")

        if song_id not in self.song_id_to_index:
            raise ValueError(f"❌ Song {song_id} not found!")

        song_idx = self.song_id_to_index[song_id]

        # Validate index
        if song_idx >= self.user_item_matrix.shape[1]:
            raise ValueError(
                f"❌ Song index {song_idx} out of bounds (max: {self.user_item_matrix.shape[1]-1})")

        # Get item factor (remember the swap!)
        item_factor = self.model.user_factors[song_idx]

        # Calculate similarity with all items
        similarities = self.model.user_factors.dot(item_factor)

        # Get top N (excluding the song itself)
        similarities[song_idx] = -np.inf
        top_indices = np.argsort(similarities)[::-1][:top_n]

        similar_songs = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                similar_song_id = self.index_to_song_id[int(idx)]
                similar_songs.append({
                    'song_id': similar_song_id,
                    'similarity_score': float(similarities[idx])
                })

        return similar_songs

    def get_song_details(self, song_id):
        """Get track details from database"""
        track = self.db.query(Track).filter(Track.id == song_id).first()
        return track

    def debug_info(self, user_id=None):
        """Print debug information."""
        print("\n🔍 DEBUG INFO:")
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"Number of users in mapping: {len(self.user_id_to_index)}")
        print(f"Number of songs in mapping: {len(self.song_id_to_index)}")

        if self.is_trained:
            print(f"Model user_factors shape: {self.model.user_factors.shape}")
            print(f"Model item_factors shape: {self.model.item_factors.shape}")
            print("Note: Due to transpose, user_factors are actually song embeddings")
            print("      and item_factors are actually user embeddings")

        if user_id:
            if user_id in self.user_id_to_index:
                idx = self.user_id_to_index[user_id]
                print(f"{user_id} index: {idx}")
                print(
                    f"Valid user index range: 0-{self.user_item_matrix.shape[0]-1}")
            else:
                print(f"❌ {user_id} NOT found in mapping!")

    def close(self):
        """Close database connection"""
        self.db.close()
