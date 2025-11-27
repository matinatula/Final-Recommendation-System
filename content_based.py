import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from database import SessionLocal, Track, SongFeature


class ContentBasedRecommender:
    """
    Content-based filtering using audio feature similarity.
    Recommends songs with similar acoustic properties (MFCC, tempo, spectral features).
    """

    def __init__(self):
        self.db = SessionLocal()
        self.feature_matrix = None
        self.song_ids = []
        self.scaler = StandardScaler()

    def load_features(self):
        """Load audio features from database and construct feature matrix."""
        print("Loading audio features from database...")

        features = self.db.query(SongFeature).all()
        if not features:
            raise ValueError("No song features found in database")

        self.song_ids = [f.song_id for f in features]

        feature_list = []
        for f in features:
            feature_vector = (
                f.mfcc +
                [f.tempo] +
                f.chroma +
                [f.spectral_centroid] +
                [f.spectral_bandwidth] +
                f.spectral_contrast +
                [f.rms_energy] +
                [f.zcr]
            )
            feature_list.append(feature_vector)

        self.feature_matrix = np.array(feature_list)
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)

        print(
            f"Loaded {len(self.song_ids)} songs with {self.feature_matrix.shape[1]} features")

    def get_similar_songs(self, song_id, top_n=10):
        """
        Find songs with similar audio characteristics.
        
        Args:
            song_id: Reference song identifier
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended songs with similarity scores
        """
        if self.feature_matrix is None:
            self.load_features()

        if song_id not in self.song_ids:
            raise ValueError(f"Song {song_id} not found in database")

        song_index = self.song_ids.index(song_id)
        song_vector = self.feature_matrix[song_index].reshape(1, -1)

        similarities = cosine_similarity(song_vector, self.feature_matrix)[0]
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'song_id': self.song_ids[idx],
                'score': float(similarities[idx])
            })

        return recommendations

    def recommend_for_user_history(self, user_song_ids, top_n=10):
        """
        Generate recommendations based on user's listening history.
        Creates a taste profile by averaging features of listened songs.
        
        Args:
            user_song_ids: List of songs the user has listened to
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended songs with scores
        """
        if self.feature_matrix is None:
            self.load_features()

        user_indices = []
        for song_id in user_song_ids:
            if song_id in self.song_ids:
                user_indices.append(self.song_ids.index(song_id))

        if not user_indices:
            raise ValueError("None of the user's songs found in database")

        user_profile = np.mean(
            self.feature_matrix[user_indices], axis=0).reshape(1, -1)
        similarities = cosine_similarity(user_profile, self.feature_matrix)[0]

        for idx in user_indices:
            similarities[idx] = -1

        similar_indices = np.argsort(similarities)[::-1][:top_n]

        recommendations = []
        for idx in similar_indices:
            if similarities[idx] > -1:
                recommendations.append({
                    'song_id': self.song_ids[idx],
                    'score': float(similarities[idx])
                })

        return recommendations

    def close(self):
        """Close database connection."""
        self.db.close()
