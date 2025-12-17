# ---- ndcg.py ----
import numpy as np
from database import SessionLocal, UserListeningHistory, Track, SongFeature
from content_based import ContentBasedRecommender
from collaborative_als import CollaborativeRecommender
from hybrid import HybridRecommender


class NDCGEvaluator:
    """Evaluates recommendation quality using NDCG@k metric."""

    def __init__(self):
        self.db = SessionLocal()

    def calculate_dcg(self, relevance_scores, k):
        """Calculate Discounted Cumulative Gain at position k."""
        relevance_scores = relevance_scores[:k]
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            position = i + 1
            dcg += rel / np.log2(position + 1)
        return dcg

    def calculate_idcg(self, relevance_scores, k):
        """Calculate Ideal DCG (best possible ranking)."""
        ideal_relevance = sorted(relevance_scores, reverse=True)
        return self.calculate_dcg(ideal_relevance, k)

    def calculate_ndcg(self, relevance_scores, k):
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Returns:
            NDCG score between 0 and 1 (1 = perfect ranking)
        """
        dcg = self.calculate_dcg(relevance_scores, k)
        idcg = self.calculate_idcg(relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def get_relevance_scores(self, user_id, recommended_song_ids):
        """
        Calculate relevance scores for recommended songs.
        
        A song is relevant if:
        1. User has listened to it before (high relevance)
        2. It's similar to songs the user has listened to (medium relevance)
        
        Args:
            user_id: User identifier
            recommended_song_ids: List of recommended song IDs
            
        Returns:
            List of relevance scores (0 to 1)
        """
        history = self.db.query(UserListeningHistory)\
            .filter(UserListeningHistory.user_id == user_id)\
            .all()

        if not history:
            return [0.0] * len(recommended_song_ids)

        user_listens = {}
        for h in history:
            user_listens[h.song_id] = h.listen_count

        all_features = {}
        features_query = self.db.query(SongFeature).all()
        for f in features_query:
            all_features[f.song_id] = {
                'tempo': f.tempo,
                'energy': f.rms_energy
            }

        relevance_scores = []
        for song_id in recommended_song_ids:
            if song_id in user_listens:
                listen_count = user_listens[song_id]
                relevance = min(1.0, np.log1p(listen_count) / 3.0)
                relevance_scores.append(relevance)
            else:
                if song_id not in all_features:
                    relevance_scores.append(0.0)
                    continue

                rec_features = all_features[song_id]
                similarities = []

                for listened_song_id in user_listens.keys():
                    if listened_song_id in all_features:
                        listened_features = all_features[listened_song_id]

                        tempo_diff = abs(
                            rec_features['tempo'] - listened_features['tempo']) / 100
                        energy_diff = abs(
                            rec_features['energy'] - listened_features['energy']) / 0.2

                        distance = tempo_diff + energy_diff
                        similarity = 1.0 / (1.0 + distance)

                        weight = np.log1p(user_listens[listened_song_id])
                        similarities.append(similarity * weight)

                if similarities:
                    avg_similarity = np.mean(similarities)
                    relevance = avg_similarity * 0.7
                    relevance_scores.append(relevance)
                else:
                    relevance_scores.append(0.0)

        return relevance_scores

    def evaluate_recommender(self, recommender, user_ids, k=10, emotion='neutral'):
        """
        Evaluate a recommender system using NDCG@k.
        
        Args:
            recommender: Recommender instance to evaluate
            user_ids: List of user IDs for evaluation
            k: Number of top recommendations to evaluate
            emotion: Target emotion (for hybrid system)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating on {len(user_ids)} users (NDCG@{k})...")

        ndcg_scores = []
        successful_evaluations = 0

        for user_id in user_ids:
            try:
                if isinstance(recommender, HybridRecommender):
                    recs = recommender.recommend(
                        user_id=user_id, target_emotion=emotion, top_n=k)
                    recommended_song_ids = [r['song_id'] for r in recs]

                elif isinstance(recommender, CollaborativeRecommender):
                    recs = recommender.recommend_for_user(
                        user_id=user_id, top_n=k)
                    recommended_song_ids = [r['song_id'] for r in recs]

                elif isinstance(recommender, ContentBasedRecommender):
                    history = self.db.query(UserListeningHistory)\
                        .filter(UserListeningHistory.user_id == user_id)\
                        .limit(20)\
                        .all()

                    if not history:
                        continue

                    user_song_ids = [h.song_id for h in history]
                    recs = recommender.recommend_for_user_history(
                        user_song_ids=user_song_ids, top_n=k)
                    recommended_song_ids = [r['song_id'] for r in recs]

                else:
                    raise ValueError("Unknown recommender type")

                relevance_scores = self.get_relevance_scores(
                    user_id, recommended_song_ids)
                ndcg = self.calculate_ndcg(relevance_scores, k)
                ndcg_scores.append(ndcg)
                successful_evaluations += 1

            except Exception:
                continue

        if ndcg_scores:
            mean_ndcg = np.mean(ndcg_scores)
            std_ndcg = np.std(ndcg_scores)
            min_ndcg = np.min(ndcg_scores)
            max_ndcg = np.max(ndcg_scores)
        else:
            mean_ndcg = std_ndcg = min_ndcg = max_ndcg = 0.0

        print(
            f"Evaluated {successful_evaluations}/{len(user_ids)} users successfully")

        return {
            'mean_ndcg': mean_ndcg,
            'std_ndcg': std_ndcg,
            'min_ndcg': min_ndcg,
            'max_ndcg': max_ndcg,
            'num_users': successful_evaluations,
            'all_scores': ndcg_scores
        }

    def compare_recommenders(self, user_ids, k=10):
        """
        Compare content-based, collaborative, and hybrid recommenders.
        
        Args:
            user_ids: List of user IDs for evaluation
            k: Number of top recommendations to evaluate
            
        Returns:
            Dictionary containing results for all systems
        """
        print("\n" + "="*60)
        print("COMPARING RECOMMENDATION SYSTEMS")
        print("="*60)

        results = {}

        print("\n[1/3] Content-Based Recommender:")
        content_rec = ContentBasedRecommender()
        content_rec.load_features()
        results['content'] = self.evaluate_recommender(
            content_rec, user_ids, k)
        content_rec.close()

        print("\n[2/3] Collaborative Filtering:")
        collab_rec = CollaborativeRecommender()
        collab_rec.load_data()
        collab_rec.train()
        results['collaborative'] = self.evaluate_recommender(
            collab_rec, user_ids, k)
        collab_rec.close()

        print("\n[3/3] Hybrid System:")
        hybrid_rec = HybridRecommender()
        hybrid_rec.load_and_train()
        results['hybrid'] = self.evaluate_recommender(
            hybrid_rec, user_ids, k, emotion='neutral')
        hybrid_rec.close()

        return results

    def print_comparison_table(self, results):
        """Print evaluation results in table format."""
        print("\n" + "="*60)
        print("NDCG EVALUATION RESULTS")
        print("="*60)

        print(
            f"\n{'System':<20} {'Mean NDCG':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8}")
        print("-" * 60)

        for system_name, metrics in results.items():
            print(f"{system_name.capitalize():<20} "
                  f"{metrics['mean_ndcg']:.4f}       "
                  f"{metrics['std_ndcg']:.4f}       "
                  f"{metrics['min_ndcg']:.4f}   "
                  f"{metrics['max_ndcg']:.4f}")

        print("\n" + "="*60)
        best_system = max(
            results.keys(), key=lambda x: results[x]['mean_ndcg'])
        best_score = results[best_system]['mean_ndcg']

        print(f"BEST SYSTEM: {best_system.upper()}")
        print(f"NDCG Score: {best_score:.4f}")

        if best_score > 0.8:
            print("Evaluation: Excellent recommendation quality")
        elif best_score > 0.6:
            print("Evaluation: Good recommendation quality")
        elif best_score > 0.4:
            print("Evaluation: Moderate recommendation quality")
        else:
            print("Evaluation: Needs improvement")

    def close(self):
        """Close database connection."""
        self.db.close()


if __name__ == "__main__":
    print("="*60)
    print("NDCG EVALUATION")
    print("="*60)

    evaluator = NDCGEvaluator()

    print("\nLoading test users...")
    db = SessionLocal()
    all_users = db.query(UserListeningHistory.user_id)\
        .distinct()\
        .limit(50)\
        .all()
    test_user_ids = [u[0] for u in all_users]
    print(f"Testing on {len(test_user_ids)} users")
    db.close()

    results = evaluator.compare_recommenders(user_ids=test_user_ids, k=10)
    evaluator.print_comparison_table(results)
    evaluator.close()

    print("\n" + "="*60)
    print("Evaluation complete")
    print("="*60)


# ---- hybrid.py ----
import numpy as np
from content_based import ContentBasedRecommender
from collaborative_als import CollaborativeRecommender
from emotion_based import EmotionBasedFilter
from database import SessionLocal, Track, UserListeningHistory


class HybridRecommender:
    """
    Hybrid recommendation system combining content-based filtering,
    collaborative filtering (ALS), and emotion-aware recommendations.
    """

    def __init__(self, content_weight=0.3, collaborative_weight=0.5, emotion_weight=0.2):
        self.w1 = content_weight
        self.w2 = collaborative_weight
        self.w3 = emotion_weight

        total = self.w1 + self.w2 + self.w3
        if not np.isclose(total, 1.0):
            self.w1 /= total
            self.w2 /= total
            self.w3 /= total

        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.emotion_filter = EmotionBasedFilter()
        self.db = SessionLocal()

        print(f"Hybrid Recommender initialized with weights: "
              f"Content={self.w1:.2f}, Collaborative={self.w2:.2f}, Emotion={self.w3:.2f}")

    def load_and_train(self):
        """Load data and train all recommender modules."""
        print("\nLoading and training hybrid system...")

        print("[1/3] Loading content-based features...")
        self.content_recommender.load_features()

        print("[2/3] Training collaborative filtering model...")
        self.collaborative_recommender.load_data()
        self.collaborative_recommender.train()

        print("[3/3] Loading emotion labels...")
        self.emotion_filter.load_emotion_labels()

        print("Hybrid system ready.\n")

    def _normalize_scores(self, recommendations):
        """Normalize scores to 0-1 range using min-max normalization."""
        if not recommendations:
            return []

        scores = [rec['score'] for rec in recommendations]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            for rec in recommendations:
                rec['score'] = 0.5
            return recommendations

        normalized = []
        for rec in recommendations:
            normalized_score = (rec['score'] - min_score) / \
                (max_score - min_score)
            normalized.append({
                'song_id': rec['song_id'],
                'score': normalized_score
            })

        return normalized

    def recommend(self, user_id, target_emotion='neutral', top_n=10):
        """
        Generate personalized recommendations for a user.
        
        Args:
            user_id: User identifier
            target_emotion: Target emotion for filtering (sad/neutral/happy/fear/angry)
            top_n: Number of recommendations to return
        
        Returns:
            List of recommended songs with scores and metadata
        """
        print(
            f"\nGenerating recommendations for {user_id} (emotion: {target_emotion})")

        # Get collaborative filtering recommendations
        try:
            collaborative_recs = self.collaborative_recommender.recommend_for_user(
                user_id=user_id,
                top_n=50
            )
        except Exception as e:
            print(f"Warning: Collaborative filtering failed - {e}")
            collaborative_recs = []

        # Get content-based recommendations
        try:
            user_history = self.db.query(UserListeningHistory)\
                .filter(UserListeningHistory.user_id == user_id)\
                .order_by(UserListeningHistory.listen_count.desc())\
                .limit(20)\
                .all()

            if user_history:
                user_song_ids = [h.song_id for h in user_history]
                content_recs = self.content_recommender.recommend_for_user_history(
                    user_song_ids=user_song_ids,
                    top_n=50
                )
            else:
                content_recs = []

        except Exception as e:
            print(f"Warning: Content-based filtering failed - {e}")
            content_recs = []

        # Normalize scores
        collaborative_recs = self._normalize_scores(collaborative_recs)
        content_recs = self._normalize_scores(content_recs)

        # Merge recommendations
        merged_scores = {}

        for rec in collaborative_recs:
            song_id = rec['song_id']
            merged_scores[song_id] = {
                'content_score': 0.0,
                'collaborative_score': rec['score'],
                'song_id': song_id
            }

        for rec in content_recs:
            song_id = rec['song_id']
            if song_id in merged_scores:
                merged_scores[song_id]['content_score'] = rec['score']
            else:
                merged_scores[song_id] = {
                    'content_score': rec['score'],
                    'collaborative_score': 0.0,
                    'song_id': song_id
                }

        # Calculate hybrid scores
        hybrid_recs = []
        for song_id, scores in merged_scores.items():
            base_score = (self.w1 * scores['content_score'] +
                          self.w2 * scores['collaborative_score'])

            hybrid_recs.append({
                'song_id': song_id,
                'score': base_score,
                'content_score': scores['content_score'],
                'collaborative_score': scores['collaborative_score']
            })

        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)

        # Apply emotion scoring
        emotion_matches = 0
        for rec in hybrid_recs:
            song_id = rec['song_id']
            song_emotion = self.emotion_filter.get_song_emotion(song_id)

            if song_emotion == target_emotion:
                emotion_score = 1.0
                emotion_matches += 1
            else:
                emotion_score = 0.0

            rec['emotion'] = song_emotion if song_emotion else 'unknown'
            rec['emotion_match'] = (song_emotion == target_emotion)
            rec['score'] = rec['score'] + (self.w3 * emotion_score)

        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)
        final_recs = hybrid_recs[:top_n]

        # Add track details
        detailed_recs = []
        for rec in final_recs:
            track = self.db.query(Track).filter(
                Track.id == rec['song_id']).first()
            if track:
                detailed_recs.append({
                    'song_id': rec['song_id'],
                    'song_name': track.name,
                    'popularity': track.popularity,
                    'duration_ms': track.duration_ms,
                    'final_score': rec['score'],
                    'content_score': rec['content_score'],
                    'collaborative_score': rec['collaborative_score'],
                    'emotion': rec['emotion'],
                    'emotion_match': rec['emotion_match']
                })

        print(f"Returned {len(detailed_recs)} recommendations "
              f"({emotion_matches} matching emotion '{target_emotion}')")

        return detailed_recs

    def update_weights(self, content_weight, collaborative_weight, emotion_weight):
        """Update recommendation weights."""
        self.w1 = content_weight
        self.w2 = collaborative_weight
        self.w3 = emotion_weight

        total = self.w1 + self.w2 + self.w3
        self.w1 /= total
        self.w2 /= total
        self.w3 /= total

        print(f"Updated weights: Content={self.w1:.2f}, "
              f"Collaborative={self.w2:.2f}, Emotion={self.w3:.2f}")

    def close(self):
        """Close all database connections."""
        self.content_recommender.close()
        self.collaborative_recommender.close()
        self.emotion_filter.close()
        self.db.close()


if __name__ == "__main__":
    print("="*60)
    print("HYBRID RECOMMENDATION SYSTEM TEST")
    print("="*60)

    hybrid = HybridRecommender(
        content_weight=0.3,
        collaborative_weight=0.5,
        emotion_weight=0.2
    )

    hybrid.load_and_train()

    test_user_id = 'user_0001'
    emotions_to_test = ['happy', 'sad', 'neutral']

    for emotion in emotions_to_test:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS FOR {test_user_id} (Emotion: {emotion})")
        print("="*60)

        recommendations = hybrid.recommend(
            user_id=test_user_id,
            target_emotion=emotion,
            top_n=10
        )

        print(f"\n{'#':<4} {'Song Name':<35} {'Score':<8} {'Content':<8} "
              f"{'Collab':<8} {'Emotion':<8} {'Match':<6}")
        print("-" * 95)

        for i, rec in enumerate(recommendations, 1):
            match_icon = "Y" if rec['emotion_match'] else "N"
            print(f"{i:<4} {rec['song_name'][:34]:<35} "
                  f"{rec['final_score']:.3f}    "
                  f"{rec['content_score']:.3f}    "
                  f"{rec['collaborative_score']:.3f}    "
                  f"{rec['emotion']:<8} {match_icon:<6}")

    hybrid.close()
    print("\nTest completed successfully.")


# ---- emotion_based.py ----
from database import SessionLocal, EmotionLabel, Track


class EmotionBasedFilter:
    """
    Emotion-aware filtering for recommendations.
    Boosts songs matching the user's current emotional state.
    
    Supported emotions: sad, neutral, happy, fear, angry
    """

    def __init__(self):
        self.db = SessionLocal()
        self.song_emotions = {}

    def load_emotion_labels(self):
        """Load emotion labels from database into memory for fast lookup."""
        print("Loading emotion labels from database...")

        labels = self.db.query(EmotionLabel).all()
        if not labels:
            raise ValueError("No emotion labels found in database")

        for label in labels:
            self.song_emotions[label.song_id] = label.emotion

        print(f"Loaded emotion labels for {len(self.song_emotions)} songs")

        emotion_counts = {}
        for emotion in self.song_emotions.values():
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count} songs")

    def get_song_emotion(self, song_id):
        """
        Get emotion label for a song.
        
        Args:
            song_id: Song identifier
            
        Returns:
            Emotion string or None if not found
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        return self.song_emotions.get(song_id, None)

    def filter_by_emotion(self, song_list, target_emotion, boost_factor=2.0):
        """
        Adjust recommendation scores based on emotion matching.
        
        Args:
            song_list: List of recommendations [{'song_id': ..., 'score': ...}, ...]
            target_emotion: User's current emotion (sad/neutral/happy/fear/angry)
            boost_factor: Multiplier for matching emotions (default: 2.0)
            
        Returns:
            List of recommendations with adjusted scores
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        emotion_adjusted_list = []

        for item in song_list:
            song_id = item['song_id']
            original_score = item['score']
            song_emotion = self.song_emotions.get(song_id, None)

            if song_emotion is None:
                emotion_match_multiplier = 1.0
            elif song_emotion == target_emotion:
                emotion_match_multiplier = boost_factor
            else:
                emotion_match_multiplier = 1.0

            adjusted_score = original_score * emotion_match_multiplier

            adjusted_item = item.copy()
            adjusted_item.update({
                'score': adjusted_score,
                'emotion': song_emotion,
                'emotion_match': (song_emotion == target_emotion)
            })

            emotion_adjusted_list.append(adjusted_item)

        emotion_adjusted_list.sort(key=lambda x: x['score'], reverse=True)

        return emotion_adjusted_list

    def get_songs_by_emotion(self, emotion, limit=100):
        """
        Retrieve songs with a specific emotion.
        
        Args:
            emotion: Target emotion
            limit: Maximum number of songs to return
            
        Returns:
            List of song IDs
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        matching_songs = [
            song_id
            for song_id, song_emotion in self.song_emotions.items()
            if song_emotion == emotion
        ]

        return matching_songs[:limit]

    def get_emotion_distribution(self, song_ids):
        """
        Calculate emotion distribution for a list of songs.
        
        Args:
            song_ids: List of song IDs
            
        Returns:
            Dictionary with emotion counts
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        emotion_counts = {
            'sad': 0,
            'neutral': 0,
            'happy': 0,
            'fear': 0,
            'angry': 0
        }

        for song_id in song_ids:
            emotion = self.song_emotions.get(song_id, 'neutral')
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1

        return emotion_counts

    def close(self):
        """Close database connection."""
        self.db.close()


# ---- database.py ----
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Track(Base):
    """Track metadata."""
    __tablename__ = "tracks"

    id = Column(Text, primary_key=True)
    name = Column(Text)
    duration_ms = Column(Integer)
    popularity = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class User(Base):
    """User account information."""
    __tablename__ = "users"

    id = Column(Text, primary_key=True)
    email = Column(Text)
    name = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class SongFeature(Base):
    """Audio features for content-based filtering."""
    __tablename__ = "song_features"

    song_id = Column(Text, primary_key=True)
    mfcc = Column(ARRAY(Float))
    tempo = Column(Float)
    chroma = Column(ARRAY(Float))
    spectral_centroid = Column(Float)
    spectral_bandwidth = Column(Float)
    spectral_contrast = Column(ARRAY(Float))
    rms_energy = Column(Float)
    zcr = Column(Float)


class UserListeningHistory(Base):
    """User listening behavior for collaborative filtering."""
    __tablename__ = "user_listening_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text)
    song_id = Column(Text)
    listen_count = Column(Integer, default=1)
    last_played_at = Column(TIMESTAMP, default=datetime.utcnow)


class EmotionLabel(Base):
    """Emotion classifications for emotion-aware filtering."""
    __tablename__ = "emotion_labels"

    song_id = Column(Text, primary_key=True)
    emotion = Column(Text)


def get_db():
    """Create and return a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


# ---- content_based.py ----
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


# ---- combine_files.py ----
import os

# Replace this with the path to your folder containing Python files
folder_path = "/Users/matinatuladhar/Desktop/final_attempt/"

# Name of the new file where all code will be combined
output_file = "combined_code.py"

# Initialize a string to hold all code
all_code = ""

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):  # Only process Python files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            # Add a header with the filename for clarity
            all_code += f"# ---- {filename} ----\n"
            all_code += code + "\n\n"

# Save all the combined code into the output file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_code)

print(
    f"All Python files in '{folder_path}' have been combined into '{output_file}'.")


# ---- collaborative_als.py ----
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


