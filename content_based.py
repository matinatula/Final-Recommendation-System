# content_based.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from database import SessionLocal, Track, SongFeature


class ContentBasedRecommender:
    """
    Content-based recommender using audio features.
    Recommends songs that sound similar based on MFCC, tempo, spectral features, etc.
    """

    def __init__(self):
        """Initialize the recommender"""
        self.db = SessionLocal()
        self.feature_matrix = None
        self.song_ids = []
        self.scaler = StandardScaler()

    def load_features(self):
        """
        Load all song features from database and prepare feature matrix.
        This creates a matrix where each row is a song, each column is a feature.
        """
        print("📥 Loading audio features from database...")

        # Query all song features from database
        features = self.db.query(SongFeature).all()

        if not features:
            raise ValueError("❌ No song features found in database!")

        # Store song IDs (to know which row corresponds to which song)
        self.song_ids = [f.song_id for f in features]

        # Build feature matrix
        feature_list = []
        for f in features:
            # Combine all features into one vector
            feature_vector = (
                f.mfcc +                    # MFCC coefficients (13 values)
                [f.tempo] +                 # Tempo (1 value)
                f.chroma +                  # Chroma (12 values)
                [f.spectral_centroid] +     # Spectral centroid (1 value)
                [f.spectral_bandwidth] +    # Spectral bandwidth (1 value)
                f.spectral_contrast +       # Spectral contrast (7 values)
                [f.rms_energy] +            # RMS energy (1 value)
                [f.zcr]                     # Zero crossing rate (1 value)
            )
            feature_list.append(feature_vector)

        # Convert to numpy array
        self.feature_matrix = np.array(feature_list)

        # Normalize features (important for fair comparison)
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)

        print(f"✅ Loaded {len(self.song_ids)} songs")
        print(f"✅ Feature matrix shape: {self.feature_matrix.shape}")

    def get_similar_songs(self, song_id, top_n=10):
        """
        Find songs similar to the given song_id.
        
        Args:
            song_id: ID of the song to find similar songs for
            top_n: Number of recommendations to return
            
        Returns:
            List of (song_id, similarity_score) tuples
        """
        # Make sure features are loaded
        if self.feature_matrix is None:
            self.load_features()

        # Find the index of the input song
        if song_id not in self.song_ids:
            raise ValueError(f"❌ Song {song_id} not found in database!")

        song_index = self.song_ids.index(song_id)

        # Get the feature vector for this song
        song_vector = self.feature_matrix[song_index].reshape(1, -1)

        # Calculate cosine similarity with ALL songs
        similarities = cosine_similarity(song_vector, self.feature_matrix)[0]

        # Get indices of most similar songs (excluding the song itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        # Build result list
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'song_id': self.song_ids[idx],
                'score': float(similarities[idx])
            })

        return recommendations

    def recommend_for_user_history(self, user_song_ids, top_n=10):
        """
        Recommend songs based on a user's listening history.
        Finds songs similar to what the user has already listened to.
        
        Args:
            user_song_ids: List of song IDs the user has listened to
            top_n: Number of recommendations to return
            
        Returns:
            List of (song_id, score) tuples
        """
        # Make sure features are loaded
        if self.feature_matrix is None:
            self.load_features()

        # Get feature vectors for all songs user has listened to
        user_indices = []
        for song_id in user_song_ids:
            if song_id in self.song_ids:
                user_indices.append(self.song_ids.index(song_id))

        if not user_indices:
            raise ValueError("❌ None of the user's songs found in database!")

        # Average the feature vectors (user's "taste profile")
        user_profile = np.mean(
            self.feature_matrix[user_indices], axis=0).reshape(1, -1)

        # Calculate similarity with all songs
        similarities = cosine_similarity(user_profile, self.feature_matrix)[0]

        # Exclude songs user has already listened to
        for idx in user_indices:
            similarities[idx] = -1  # Set to -1 so they won't be recommended

        # Get top N similar songs
        similar_indices = np.argsort(similarities)[::-1][:top_n]

        # Build result list
        recommendations = []
        for idx in similar_indices:
            if similarities[idx] > -1:  # Skip songs user already heard
                recommendations.append({
                    'song_id': self.song_ids[idx],
                    'score': float(similarities[idx])
                })

        return recommendations

    def get_song_details(self, song_id):
        """
        Get track details from database.
        
        Args:
            song_id: ID of the song
            
        Returns:
            Track object with song details
        """
        track = self.db.query(Track).filter(Track.id == song_id).first()
        return track

    def close(self):
        """Close database connection"""
        self.db.close()
