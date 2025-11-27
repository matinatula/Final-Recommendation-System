import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from database import SessionLocal, UserListeningHistory, Track


class CollaborativeRecommender:
    """
    Collaborative filtering using Alternating Least Squares (ALS).
    Learns latent user preferences from implicit listening feedback.
    """

    def __init__(self, factors=50, regularization=0.01, iterations=20):
        self.db = SessionLocal()
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )

        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_id_to_index = {}
        self.index_to_user_id = {}
        self.song_id_to_index = {}
        self.index_to_song_id = {}
        self.is_trained = False

    def load_data(self):
        """Load listening history and construct user-item interaction matrix."""
        print("Loading listening history from database...")

        history = self.db.query(UserListeningHistory).all()
        if not history:
            raise ValueError("No listening history found in database")

        print(f"Found {len(history)} listening records")

        unique_users = sorted(list(set([h.user_id for h in history])))
        unique_songs = sorted(list(set([h.song_id for h in history])))

        print(f"Users: {len(unique_users)}, Songs: {len(unique_songs)}")

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

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_songs)),
            dtype=np.float32
        )

        print(f"Built user-item matrix: {self.user_item_matrix.shape}")

    def train(self):
        """Train the ALS model on the user-item matrix."""
        if self.user_item_matrix is None:
            raise ValueError("Must load data before training")

        print("Training ALS model...")

        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        self.model.fit(self.item_user_matrix)
        self.is_trained = True

        print("Model training complete")

    def recommend_for_user(self, user_id, top_n=10):
        """
        Generate recommendations for a user based on collaborative filtering.
        
        Args:
            user_id: User identifier
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended songs with scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        if user_id not in self.user_id_to_index:
            raise ValueError(f"User {user_id} not found")

        user_idx = self.user_id_to_index[user_id]

        if user_idx >= self.user_item_matrix.shape[0]:
            raise ValueError(f"User index out of bounds")

        user_row = self.user_item_matrix[user_idx]
        user_songs = set(user_row.nonzero()[1])

        # Note: Due to matrix transpose, factor arrays are swapped
        user_factors = self.model.item_factors[user_idx]
        item_factors = self.model.user_factors

        scores = item_factors.dot(user_factors)

        for song_idx in user_songs:
            scores[song_idx] = -np.inf

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
        """Find users with similar listening patterns."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if user_id not in self.user_id_to_index:
            raise ValueError(f"User {user_id} not found")

        user_idx = self.user_id_to_index[user_id]

        if user_idx >= self.user_item_matrix.shape[0]:
            raise ValueError("User index out of bounds")

        user_factor = self.model.item_factors[user_idx]
        similarities = self.model.item_factors.dot(user_factor)

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
        """Find songs with similar user engagement patterns."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if song_id not in self.song_id_to_index:
            raise ValueError(f"Song {song_id} not found")

        song_idx = self.song_id_to_index[song_id]

        if song_idx >= self.user_item_matrix.shape[1]:
            raise ValueError("Song index out of bounds")

        item_factor = self.model.user_factors[song_idx]
        similarities = self.model.user_factors.dot(item_factor)

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

    def close(self):
        """Close database connection."""
        self.db.close()
