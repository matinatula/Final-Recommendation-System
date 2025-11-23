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
